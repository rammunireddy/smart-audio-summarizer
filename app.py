"""
Local Video Transcriber - 100% Offline (v4)
==========================================
Powered by faster-whisper + Gradio 6.x.

Features:
- Multi-file batch upload (process multiple videos at once)
- Streaming per-segment progress with ETA
- Transcript-only output (.txt)
- Cancel button to stop running jobs
- large-v3 model default (best for Hindi/Telugu)
- Also supports .mp3 and .wav audio files
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
from faster_whisper import WhisperModel

# Ensure the local directory is in PATH so ffmpeg.exe is found
os.environ["PATH"] += os.pathsep + str(Path(__file__).parent.resolve())


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "transcriptions"
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED = [
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".flv", ".wmv", ".m4v", ".mp3", ".wav",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HELPERS & SECURITY
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Security: Remove potentially dangerous characters from filenames."""
    # Keep only alphanumeric, underscores, and dashes
    name = Path(name).stem
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[-\s]+", "_", name)
    return name


def get_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = seconds
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    secs = int(td % 60)
    milliseconds = int((td % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_as_srt(segments) -> str:
    """Format Whisper segments into a standard .srt file content."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        if seg.text.strip():
            start = get_srt_timestamp(seg.start)
            end = get_srt_timestamp(seg.end)
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(srt_lines)


def check_ffmpeg() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except Exception:
        return False


def extract_audio(video_path: str, wav_path: str) -> None:
    """ffmpeg: extract 16 kHz mono WAV from video/audio."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg error:\n{r.stderr[-2000:]}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found!\n"
            "Download: https://ffmpeg.org/download.html\n"
            "Add its bin/ folder to Windows PATH and restart the app."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out (>10 min).")


def get_video_path(video_file) -> str:
    """
    Normalize the Gradio file upload value.
    Gradio 6.x gr.File returns either:
      - a plain str path
      - a NamedString / object with .name attribute
      - a dict with key 'name' or 'path'
    """
    if video_file is None:
        return None
    if isinstance(video_file, str):
        return video_file
    if isinstance(video_file, dict):
        return video_file.get("path") or video_file.get("name", "")
    if hasattr(video_file, "name"):
        return video_file.name
    return str(video_file)


# ---------------------------------------------------------------------------
# GLOBAL CACHE
# ---------------------------------------------------------------------------

CURRENT_MODEL = None
CURRENT_MODEL_PARAMS = None
CURRENT_REFINER = None

def get_model(model_size: str):
    """Load model once and cache it. Reloads only if settings change."""
    global CURRENT_MODEL, CURRENT_MODEL_PARAMS
    import torch
    import gc

    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"

    params = (model_size, device, compute_type)

    if CURRENT_MODEL is None or CURRENT_MODEL_PARAMS != params:
        # Free up memory before loading new model
        CURRENT_MODEL = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        CURRENT_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
        CURRENT_MODEL_PARAMS = params

    return CURRENT_MODEL, device


def get_refiner_model():
    """Load a tiny LLM (Qwen 0.5B) for text refinement."""
    global CURRENT_REFINER
    if CURRENT_REFINER is not None:
        return CURRENT_REFINER

    from transformers import pipeline
    import torch

    # Use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        CURRENT_REFINER = pipeline(
            "text-generation",
            model=model_id,
            device=device,
            torch_dtype="auto",
        )
    except Exception as e:
        print(f"Refiner Load Error: {e}")
        return None
    
    return CURRENT_REFINER


def _run_llm_task(pipe, text: str, mode: str = "refine") -> str:
    """Internal helper to run the LLM with specific parameters to prevent looping."""
    if mode == "summarize":
        system_msg = "You are a professional secretary. Summarize the following transcript into a main title and 5-7 concise bullet points. Focus on the main goal, key actions, and results. Return ONLY the summary."
    elif mode == "summarize_partial":
        system_msg = "You are an assistant. Extract the 3-5 most important points from this section of a transcript. Be extremely brief. Return ONLY the points."
    elif mode == "summarize_final":
        system_msg = "You are a professional secretary. Combine the following partial summaries into one cohesive final summary with a Title and 5-10 bullet points. Return ONLY the final summary."
    else:
        system_msg = "You are a professional editor. Clean up the following transcript by fixing grammar, spelling, and punctuation. Maintain the original language and meaning. Remove filler words. Return ONLY the corrected text."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Text:\n{text}"},
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Small models like Qwen-0.5B NEED repetition_penalty and a bit of temperature
    outputs = pipe(
        prompt,
        max_new_tokens=200 if mode == "summarize_partial" else (600 if "summarize" in mode else max(len(text.split()), 100) + 50),
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.2,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    
    # Extract response after the assistant prompt
    result = outputs[0]["generated_text"].split("assistant\n")[-1].strip()
    return result


def refine_text(text: str, mode: str = "refine") -> str:
    """Uses a tiny LLM to clean up or summarize the transcript with chunking support."""
    pipe = get_refiner_model()
    if not pipe:
        return text
    
    # Chunking logic for long transcripts (0.5B models struggle with > 2000 words)
    words = text.split()
    chunk_size = 2000
    
    if mode == "summarize" and len(words) > chunk_size:
        chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        partial_summaries = []
        for chunk in chunks:
            partial = _run_llm_task(pipe, chunk, mode="summarize_partial")
            partial_summaries.append(partial)
        
        combined_summaries = "\n".join(partial_summaries)
        return _run_llm_task(pipe, combined_summaries, mode="summarize_final")
    
    return _run_llm_task(pipe, text, mode=mode)


# ---------------------------------------------------------------------------
# CORE TRANSCRIPTION PIPELINE
# ---------------------------------------------------------------------------

def transcribe_batch(video_files, model_size: str, do_refine: bool, do_summary: bool, do_translate: bool, progress=gr.Progress()):
    """
    Batch pipeline: validate all files -> load models -> loop files
    sequentially. Each file: extract audio -> transcribe/translate -> refine -> summary -> save .txt/.srt.
    """
    tmp_dirs = []
    wall_start = time.time()

    try:
        # -- 0. Normalise input ------------------------------------------------
        if video_files is None:
            raise gr.Error("Please upload at least one video file.")
        if not isinstance(video_files, list):
            video_files = [video_files]

        paths = [get_video_path(f) for f in video_files]
        paths = [p for p in paths if p and os.path.isfile(p)]
        if not paths:
            raise gr.Error("No valid video files found. Please re-upload.")

        for p in paths:
            ext = Path(p).suffix.lower()
            if ext not in SUPPORTED:
                raise gr.Error(f"Unsupported format '{ext}' in {Path(p).name}")

        if not check_ffmpeg():
            raise gr.Error("ffmpeg not found on PATH.")

        total_files = len(paths)
        progress(0.02, desc="[1/5] Preparing Whisper model...")

        # -- 1. Load Whisper ---------------------------------------------------
        model, device = get_model(model_size)
        
        # -- 2. Pre-load Refiner if needed -------------------------------------
        if do_refine or do_summary:
            progress(0.05, desc="[2/5] Preparing AI Refiner model...")
            get_refiner_model()

        progress(0.08, desc=f"Models ready! (Using {device.upper()}) Starting batch...")

        all_results = []   # (filename, transcript_text, summary, elapsed, info)
        saved_paths = []

        # -- 3. Process each file ----------------------------------------------
        BAND_START = 0.08
        BAND_END   = 0.92
        band_size  = (BAND_END - BAND_START) / total_files

        for file_idx, video_path in enumerate(paths):
            fname = Path(video_path).name
            file_label = f"[{file_idx + 1}/{total_files}] {fname}"
            f_base = BAND_START + file_idx * band_size

            # -- Extract audio -------------------------------------------------
            progress(f_base, desc=f"{file_label} - extracting audio...")
            tmp_dir = tempfile.mkdtemp(prefix="transcriber_")
            tmp_dirs.append(tmp_dir)
            wav_path = os.path.join(tmp_dir, "audio.wav")
            extract_audio(video_path, wav_path)

            # -- Transcribe/Translate ------------------------------------------
            t0 = time.time()
            task_type = "translate" if do_translate else "transcribe"
            segments_gen, info = model.transcribe(
                wav_path,
                beam_size=5,
                task=task_type,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            total_dur = info.duration or 0
            segments  = []
            SEG_START = f_base + band_size * 0.1
            SEG_END   = f_base + band_size * 0.7

            for segment in segments_gen:
                segments.append(segment)
                if total_dur > 0:
                    frac     = min(segment.end / total_dur, 1.0)
                    prog_val = SEG_START + frac * (SEG_END - SEG_START)
                else:
                    prog_val = SEG_START

                elapsed_so_far = time.time() - t0
                speed = segment.end / elapsed_so_far if elapsed_so_far > 0 else 1.0
                progress(prog_val, desc=f"{file_label} - Hearing audio... ({speed:.1f}x)")

            raw_transcript = " ".join([s.text.strip() for s in segments])
            
            # -- AI Refinement -------------------------------------------------
            final_text = raw_transcript
            if do_refine:
                progress(f_base + band_size * 0.75, desc=f"{file_label} - ✨ Polishing text...")
                final_text = refine_text(raw_transcript, mode="refine")
            
            # -- AI Summary ----------------------------------------------------
            summary_text = ""
            if do_summary:
                progress(f_base + band_size * 0.85, desc=f"{file_label} - 📝 Writing summary...")
                # Use final_text (refined) if available to speed up summary
                summary_text = refine_text(final_text, mode="summarize")

            # -- Generate SRT --------------------------------------------------
            srt_content = format_as_srt(segments)

            elapsed_file = time.time() - t0

            # -- Save Files (Security: Sanitize names) -------------------------
            safe_name = sanitize_filename(fname)
            
            txt_path = OUTPUT_DIR / f"{safe_name}_transcript.txt"
            txt_path.write_text(final_text, encoding="utf-8")
            
            srt_path = OUTPUT_DIR / f"{safe_name}.srt"
            srt_path.write_text(srt_content, encoding="utf-8")
            
            if summary_text:
                sum_path = OUTPUT_DIR / f"{safe_name}_summary.txt"
                sum_path.write_text(summary_text, encoding="utf-8")

            saved_paths.append(txt_path)
            all_results.append((fname, final_text, summary_text, elapsed_file, info))

            progress(f_base + band_size, desc=f"{file_label} done!")

        progress(0.93, desc="Wrapping up...")

        # -- 4. Combined Result for UI -----------------------------------------
        full_display = ""
        for fn, txt, summary, *_ in all_results:
            full_display += f"=== {fn} ===\n"
            if summary:
                full_display += f"SUMMARY:\n{summary}\n\n"
            full_display += f"TRANSCRIPT:\n{txt}\n\n" + ("="*40) + "\n\n"

        result_md = f"### {total_files} file(s) processed!\nFiles saved in: `{OUTPUT_DIR}`"
        
        return (
            result_md,
            f"Transcripts, Subtitles, and Summaries saved to `{OUTPUT_DIR}`.",
            full_display,
            gr.update(value=str(saved_paths[0]), visible=True),
        )

        progress(0.93, desc="Wrapping up...")

        # -- 3. Combined file (if multiple) ------------------------------------
        if total_files > 1:
            combined_parts = []
            for fn, txt, *_ in all_results:
                combined_parts.append(f"=== {fn} ===\n{txt}")
            combined_text = "\n\n".join(combined_parts)
            combined_path = OUTPUT_DIR / "batch_combined_transcript.txt"
            combined_path.write_text(combined_text, encoding="utf-8")
            download_path = combined_path
            display_text  = combined_text
        else:
            download_path = saved_paths[0]
            display_text  = all_results[0][1]

        # -- 4. Summary panel --------------------------------------------------
        wall_total = time.time() - wall_start
        rows = []
        for fn, txt, elapsed, finfo, segs in all_results:
            lang  = finfo.language.upper()
            dur   = finfo.duration or 0
            words = len(txt.split())
            rows.append(
                f"| `{fn}` | {lang} | "
                f"{int(dur // 60)}m{int(dur % 60):02d}s | "
                f"{elapsed:.0f}s | ~{words:,} |"
            )

        result_md = (
            f"### {total_files} file(s) transcribed!\n\n"
            "| File | Lang | Duration | Time | Words |\n"
            "|------|------|----------|------|-------|\n"
            + "\n".join(rows)
            + f"\n\n**Total time:** {wall_total:.0f}s  |  "
            f"**Model:** `{model_size}`"
        )

        save_rows = "\n".join(f"- `{p.name}`" for p in saved_paths)
        if total_files > 1:
            save_rows += "\n- `batch_combined_transcript.txt` _(all combined)_"
        save_md = (
            f"### Saved to:\n`{OUTPUT_DIR}`\n\n" + save_rows
        )

        progress(1.0, desc="All done!")

        return (
            result_md,
            save_md,
            display_text,
            gr.update(value=str(download_path), visible=True),
        )

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Error: {e}\n\n{traceback.format_exc()}")
    finally:
        for d in tmp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------------------------

CSS = """
body, .gradio-container {
    background: #0a0a0f !important;
    color: #e2e8f0 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f3460 100%);
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    margin-bottom: 16px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
.hero h1 {
    font-size: 2.4em;
    font-weight: 900;
    margin: 0 0 8px 0;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #64748b;
    font-size: 1.05em;
    margin: 0 0 16px 0;
}
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin: 3px;
}
button.cancel-btn {
    background: linear-gradient(135deg, #dc2626, #991b1b) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(220,38,38,0.35) !important;
    transition: all 0.2s ease !important;
}
button.cancel-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(220,38,38,0.5) !important;
}
button.lg.primary {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 16px 32px !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.45) !important;
    transition: all 0.2s ease !important;
}
button.lg.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(37,99,235,0.6) !important;
}
.result-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px 16px 16px 16px;
    margin-top: 12px;
}
.save-box {
    background: #0f2a1a;
    border: 1px solid #14532d;
    border-radius: 12px;
    padding: 24px 16px 16px 16px;
    margin-top: 12px;
}
.tab-nav button {
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
}
.tab-nav button.selected {
    color: #60a5fa !important;
    border-bottom: 3px solid #60a5fa !important;
}
textarea {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    font-family: Consolas, monospace !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
    border-radius: 10px !important;
}
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #1e293b; padding: 8px 12px; }
th { background: #1e293b; color: #93c5fd; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
"""

HERO_HTML = """
<div class="hero">
  <h1>Local Video Transcriber</h1>
  <p>100% Offline  |  Powered by Whisper AI + Qwen LLM  |  No data leaves your machine</p>
  <span class="badge" style="background:#1e3a5f;color:#60a5fa;">faster-whisper</span>
  <span class="badge" style="background:#14532d;color:#34d399;">Qwen-0.5B AI</span>
  <span class="badge" style="background:#3b0764;color:#a78bfa;">Auto Language</span>
  <span class="badge" style="background:#450a0a;color:#f87171;">100% Private</span>
</div>
"""


THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="violet",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0a0a0f",
    block_background_fill="#0d1117",
    block_border_color="#1e293b",
    input_background_fill="#0d1117",
)


def build_ui():
    with gr.Blocks(
        title="Local Video Transcriber - 100% Offline",
    ) as demo:

        gr.HTML(HERO_HTML)

        with gr.Tabs():

            # == TAB 1 - Upload & Run ==========================================
            with gr.Tab("Upload & Transcribe"):
                with gr.Row(equal_height=False):

                    with gr.Column(scale=3, min_width=340):
                        gr.Markdown("### Select Your Video(s)")
                        video_input = gr.File(
                            label=(
                                "Select one or more videos "
                                "- or drag them here"
                            ),
                            file_count="multiple",
                            file_types=SUPPORTED,
                        )

                    with gr.Column(scale=2, min_width=240):
                        gr.Markdown("### Settings")
                        model_selector = gr.Dropdown(
                            label="Whisper Model",
                            choices=[
                                "tiny", "base", "small",
                                "medium", "large-v3",
                            ],
                            value="large-v3",
                        )
                        with gr.Group():
                            do_refine = gr.Checkbox(
                                label="✨ AI Refine (Fix Grammar/Spelling)",
                                value=False,
                                info="Polishes text using Qwen-0.5B."
                            )
                            do_summary = gr.Checkbox(
                                label="📝 AI Summary",
                                value=False,
                                info="Generates bullet-point takeaways."
                            )
                            do_translate = gr.Checkbox(
                                label="🌍 Translate to English",
                                value=False,
                                info="Converts any language into English."
                            )
                        gr.Markdown(
                            "| Model | Speed | Quality | Size |\n"
                            "|-------|-------|---------|------|\n"
                            "| `tiny`     | Fast x4 | Fair    | ~75 MB |\n"
                            "| `base`     | Fast x3 | Good    | ~145 MB |\n"
                            "| `small`    | Fast x2 | Great   | ~465 MB |\n"
                            "| `medium`   | Fast    | Excellent | ~769 MB |\n"
                            "| `large-v3` | Slow    | Best    | ~1.5 GB |\n\n"
                            "_`large-v3` is best for Hindi & Telugu_"
                        )
                        gr.Markdown(
                            f"**Output folder:**\n\n`{OUTPUT_DIR}`\n\n"
                            "_Files are auto-saved here._"
                        )

                with gr.Row():
                    transcribe_btn = gr.Button(
                        "Start Transcription",
                        variant="primary",
                        size="lg",
                        scale=4,
                    )
                    cancel_btn = gr.Button(
                        "Cancel",
                        variant="stop",
                        size="lg",
                        scale=1,
                        elem_classes="cancel-btn",
                    )

                gr.Markdown(
                    "> **Progress updates in real time.** "
                    "If you change the model or want to restart - "
                    "click **Cancel** first, then upload again and "
                    "click **Start Transcription**."
                )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=1):
                        result_info = gr.Markdown(
                            value=(
                                "**Status:** Waiting for video...\n\n"
                                "_Results will appear here after "
                                "transcription._"
                            ),
                            elem_classes="result-box",
                        )
                    with gr.Column(scale=1):
                        save_info = gr.Markdown(
                            value=(
                                f"**Output folder:**\n\n`{OUTPUT_DIR}`\n\n"
                                "_Transcript files saved here "
                                "automatically._"
                            ),
                            elem_classes="save-box",
                        )

            # == TAB 2 - Transcript ============================================
            with gr.Tab("Transcript (.txt)"):
                gr.Markdown("### Full Transcript Text")
                transcript_output = gr.Textbox(
                    label="Transcript",
                    placeholder=(
                        "Transcript will appear here after processing..."
                    ),
                    lines=25,
                    max_lines=80,
                    interactive=False,
                )
                txt_download = gr.File(
                    label="Download Transcript (.txt)",
                    visible=False,
                )

        gr.HTML(
            '<div style="text-align:center;color:#334155;font-size:12px;'
            'margin-top:18px;padding:10px;">'
            "Local Video Transcriber  |  "
            "faster-whisper + Gradio  |  100% local &amp; private"
            "</div>"
        )

        # -- Event handler -----------------------------------------------------
        def on_click(video_files, model_size, do_refine, do_summary, do_translate, progress=gr.Progress()):
            if video_files is None:
                raise gr.Error("Please upload at least one video file.")
            return transcribe_batch(video_files, model_size, do_refine, do_summary, do_translate, progress)

        click_event = transcribe_btn.click(
            fn=on_click,
            inputs=[video_input, model_selector, do_refine, do_summary, do_translate],
            outputs=[
                result_info,
                save_info,
                transcript_output,
                txt_download,
            ],
            show_progress="full",
            concurrency_limit=1,
        )

        # Cancel button: instantly kills the running server-side job
        cancel_btn.click(fn=None, cancels=[click_event])

    return demo


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("  Local Video Transcriber - 100% Offline  (v4)")
    print("=" * 64)
    print(f"  Output -> {OUTPUT_DIR}")

    if not check_ffmpeg():
        print("\n  WARNING: ffmpeg NOT found on PATH!")
        print("  https://ffmpeg.org/download.html")
        print("  Add its bin/ to Windows PATH, then restart.\n")
    else:
        print("  ffmpeg detected")
    print("  faster-whisper ready")
    print("  Gradio ready\n")
    print("  http://127.0.0.1:7860\n")

    app = build_ui()
    app.queue(
        max_size=1,
        default_concurrency_limit=1,
    )
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        max_threads=2,
        theme=THEME,
        css=CSS,
    )
