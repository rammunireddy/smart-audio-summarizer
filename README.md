# Local Video Transcriber - 100% Offline (Pro)

Transcribe, translate, and summarize videos locally using Whisper AI and Qwen LLM. No data ever leaves your machine.

## ✨ New in v5
- **AI Refinement**: Fixes grammar and spelling using a tiny local LLM (Qwen-0.5B).
- **AI Summarization**: Automatically generates bullet-point takeaways.
- **Subtitles (.srt)**: Generates standard subtitle files for your videos.
- **Translation**: Translate any foreign language audio directly into English.
- **Security**: Filename sanitization and local-only execution.

## Features
- Multi-file batch upload (drag & drop multiple videos)
- Models: tiny, base, small, medium, large-v3
- Auto language detection
- Real-time progress with speed and ETA
- 100% private - nothing leaves your machine

## Prerequisites
1. **Python 3.11+**
2. **ffmpeg** - must be on system PATH
   - Download: [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add `bin/` folder to Windows PATH

## Setup & Run

```bash
cd "d:\audio trastipn"
pip install -r requirements.txt
python app.py
```

Opens automatically at: `http://127.0.0.1:7860`

## Output
Files are saved to: `d:\audio trastipn\transcriptions\`
- `_transcript.txt`: Plain text transcript.
- `.srt`: Subtitle file.
- `_summary.txt`: AI-generated summary (if enabled).

## Models
| Model | Speed | Quality | Size | Best For |
|-------|-------|---------|------|----------|
| tiny | Fast x4 | Fair | ~75 MB | Quick drafts |
| base | Fast x3 | Good | ~145 MB | Short clips |
| small | Fast x2 | Great | ~465 MB | General use |
| medium | Fast | Excellent | ~769 MB | Good balance |
| large-v3 | Slow | Best | ~1.5 GB | Hindi/Telugu |

_Note: AI Refinement uses the Qwen-0.5B model (~900MB), downloaded automatically on first use._
