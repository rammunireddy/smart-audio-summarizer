# Local Video Transcriber - 100% Offline

Transcribe videos to text locally using Whisper AI. No internet needed (after first model download).

## Features
- Multi-file batch upload (drag & drop multiple videos)
- Models: tiny, base, small, medium, large-v3
- Auto language detection (English, Hindi, Telugu, etc.)
- Real-time progress with speed and ETA
- Cancel button to stop anytime
- Transcript saved as .txt automatically
- 100% private - nothing leaves your machine

## Prerequisites
1. **Python 3.11+**
2. **ffmpeg** - must be on system PATH
   - Download: https://ffmpeg.org/download.html
   - Add `bin/` folder to Windows PATH

## Setup & Run

```bash
cd "d:\audio trastipn"
pip install -r requirements.txt
python app.py
```

Opens automatically at: http://127.0.0.1:7860

## Output
Transcripts are saved to: `d:\audio trastipn\transcriptions\`

## Models
| Model | Speed | Quality | Size | Best For |
|-------|-------|---------|------|----------|
| tiny | Fast x4 | Fair | ~75 MB | Quick drafts |
| base | Fast x3 | Good | ~145 MB | Short clips |
| small | Fast x2 | Great | ~465 MB | General use |
| medium | Fast | Excellent | ~769 MB | Good balance |
| large-v3 | Slow | Best | ~1.5 GB | Hindi/Telugu |

First run downloads the selected model automatically.
