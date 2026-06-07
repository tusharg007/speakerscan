---
title: SpeakerScan
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

<p align="center">
  <h1 align="center">🎙️ SpeakerScan</h1>
  <p align="center">
    <strong>Production-grade speech data annotation pipeline</strong><br>
    Speaker Diarization · Emotion Classification · Language Identification
  </p>
  <p align="center">
    <a href="https://huggingface.co/spaces/champTUSHARg007/speakerscan">
      <img src="https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue" alt="HF Spaces">
    </a>
    <a href="https://github.com/tusharg007/speakerscan">
      <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub">
    </a>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/License-Portfolio-green" alt="License">
  </p>
</p>

---

## What is SpeakerScan?

SpeakerScan is an **end-to-end speech data annotation pipeline** that takes raw audio (uploaded files or YouTube URLs) and produces structured, per-segment annotations with:

- **Who** is speaking (speaker diarization via [pyannote](https://github.com/pyannote/pyannote-audio))
- **What emotion** they're expressing (classification via [wav2vec2](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition))
- **What language** they're speaking (identification via [OpenAI Whisper](https://github.com/openai/whisper))

Built as a production-quality data pipeline for **TTS/dubbing model training** — the kind of annotation infrastructure that powers multilingual speech synthesis at scale.

### Why This Matters

Modern TTS and dubbing systems (like those at Sarvam AI, ElevenLabs, or Coqui) need **training data annotated with speaker boundaries, emotion labels, and language tags**. Doing this manually is prohibitively expensive at scale. SpeakerScan automates the entire annotation workflow — from raw YouTube content to structured JSON/CSV — in a single command or a web UI click.

---

## 🚀 Live Demo

**Try it now:** [huggingface.co/spaces/champTUSHARg007/speakerscan](https://huggingface.co/spaces/champTUSHARg007/speakerscan)

Upload any audio file or paste a YouTube URL. The pipeline will:
1. Convert to 16kHz mono WAV
2. Diarize speakers with pyannote 3.1
3. Classify emotion per segment with wav2vec2
4. Detect language per segment with Whisper
5. Show interactive results with a Plotly timeline

> **Note:** First run downloads ~1.5 GB of model weights. Subsequent runs reuse cached models.

---

## 🏗️ Architecture

```
Input Sources                    Pipeline Stages                   Outputs
─────────────                    ───────────────                   ───────
                                 ┌─────────────┐
YouTube URLs  ─┐                 │  Download    │
               ├──► ffmpeg ───►  │  16kHz mono  │
Audio Upload  ─┘    convert      │     WAV      │
                                 └──────┬───────┘
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │  Diarize     │──► RTTM files
                                 │ (pyannote    │    (NIST standard)
                                 │  3.1)        │
                                 └──────┬───────┘
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │  Emotion     │
                                 │ (wav2vec2    │
                                 │  XLSR)       │
                                 └──────┬───────┘
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │  Language ID │
                                 │ (Whisper     │
                                 │  tiny)       │
                                 └──────┬───────┘
                                        │
                                        ▼
                                 ┌─────────────┐     ┌─── JSON (per-file)
                                 │  Annotate    │─────┤
                                 │  & Merge     │     └─── CSV  (manifest)
                                 └─────────────┘
```

### Streamlit Web Interface

```
┌──────────────────────────────────────────────────────────┐
│  🎙️ SpeakerScan                                         │
│  ┌──────────────┐  ┌──────────────┐                      │
│  │ Upload Audio │  │ YouTube URL  │  ← Two input modes   │
│  └──────┬───────┘  └──────┬───────┘                      │
│         └────────┬────────┘                              │
│                  ▼                                        │
│  ┌────────────────────────────────────┐                  │
│  │  Live Progress Bar                 │  ← Real-time     │
│  │  ████████████░░░░░░░░░░ 65%       │    stage updates  │
│  │  ✓ diarize (12.3s)               │                    │
│  │  ▶ emotion...                     │                    │
│  └────────────────────────────────────┘                  │
│                  ▼                                        │
│  ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐           │
│  │1:45  │ │ 3    │ │ English  │ │ neutral  │ ← Metrics │
│  │Dur.  │ │Spkrs │ │Language  │ │Emotion   │            │
│  └──────┘ └──────┘ └──────────┘ └──────────┘            │
│                                                          │
│  ┌────────────────────────────────────┐                  │
│  │  ████ SPEAKER_00  ████  ████      │  ← Plotly        │
│  │    ██████ SPEAKER_01 ████         │    Timeline       │
│  └────────────────────────────────────┘                  │
│                                                          │
│  ┌────────────────────────────────────┐                  │
│  │  Start │ End │ Speaker │ Emotion  │  ← Segment       │
│  │  0:03  │0:12 │ SPK_00  │ neutral  │    Table          │
│  └────────────────────────────────────┘                  │
│  [📥 Download JSON]  [📥 Download CSV]                   │
└──────────────────────────────────────────────────────────┘
```

---

## 🧠 Key Design Decisions

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| **Stage-level checkpointing** | If emotion classification crashes after 20-min diarization, resume from where it stopped — don't re-run everything | Binary done/not-done flag (forces full re-run) |
| **Immutable segment data** | Each stage returns a *new* list. If language detection fails halfway, emotion data is untouched | In-place mutation (corrupts upstream data on partial failure) |
| **Thread-safe inference locks** | `threading.Lock()` around all model `.forward()` calls. Safe for `--workers 2` without multiprocessing overhead | Process-based parallelism (4x memory cost, IPC complexity) |
| **Atomic file writes** | Write to `.tmp` then `os.replace()`. Prevents corrupt JSON/CSV if process dies mid-write | Direct writes (corrupt files on crash — invisible until a reviewer runs your code) |
| **Lazy singleton models** | Load pyannote + wav2vec2 + Whisper once, reuse across all files. Never reload per-file | Per-file loading (OOM after 3 files on 16 GB RAM) |
| **CPU-first deployment** | HF Spaces free tier has no GPU. Pipeline works on CPU; CUDA is auto-detected when available | GPU-required (blocks free-tier deployment) |

---

## 🔬 Engineering Challenges & Solutions

### Challenge 1: Three ML Models in 16 GB RAM

**Problem:** pyannote (speaker diarization), wav2vec2 (emotion), and Whisper (language) collectively need ~4 GB of model weights. On HF Spaces free tier (16 GB total), this leaves limited headroom for audio processing.

**Solution:**
- Lazy singleton loading — models are initialized once and cached for the lifetime of the app
- Whisper `tiny` variant (39M params) instead of `base` or `small` — sufficient for language ID
- Audio segments are processed sequentially, never loaded entirely into memory
- `tempfile.mkdtemp()` for all intermediate files — auto-cleaned by OS

### Challenge 2: Thread Safety Without Deadlocks

**Problem:** Streamlit reruns the entire script on each user interaction. Long-running ML inference (minutes) would freeze the UI. But PyTorch models are NOT thread-safe for concurrent forward passes.

**Solution:**
```python
# Each model module has two locks:
_model_lock = threading.Lock()      # Guards lazy initialization (one-time)
_inference_lock = threading.Lock()  # Serializes forward() calls (per-inference)

# Double-checked locking pattern prevents redundant model loads:
def get_model():
    if _model is not None:     # Fast path — no lock needed
        return _model
    with _model_lock:
        if _model is not None: # Re-check after acquiring lock
            return _model
        _model = load_model()  # Expensive — happens exactly once
    return _model
```
Pipeline runs in a background `threading.Thread`, progress updates flow through a `queue.Queue`, and the Streamlit main thread polls with `time.sleep(0.5)`.

### Challenge 3: Cross-Platform FFmpeg Resolution

**Problem:** FFmpeg is required for audio conversion, but:
- HF Spaces installs it via `packages.txt` (system-level apt)
- Windows users might not have it on PATH
- Docker containers need explicit `apt-get install`

**Solution:** A priority-based resolver that checks multiple sources:
```python
def _resolve_ffmpeg():
    # 1. System PATH (Linux/Mac/Docker — most reliable)
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    # 2. imageio-ffmpeg bundled binary (Windows fallback)
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()
```

### Challenge 4: Idempotent Pipeline with Crash Recovery

**Problem:** Processing 100 YouTube videos takes hours. If the machine crashes at video #67, how do you resume without re-processing videos 1-66?

**Solution:** A JSON checkpoint file tracks 5 stages per file:
```json
{
  "video_id_abc": {
    "status": "in_progress",
    "stage": "emotion",        // Last completed stage
    "started_at": "2026-06-07T10:00:00Z"
  }
}
```
On restart, the pipeline reads the checkpoint, sees that `download` and `diarize` are done, loads the RTTM from disk, and resumes from `emotion`. Atomic writes (`write .tmp → rename`) prevent checkpoint corruption.

### Challenge 5: Immutable Data Flow for Partial Failures

**Problem:** If emotion classification succeeds for 47 segments but language detection crashes at segment #23, what happens to the data?

**Solution:** Each enrichment stage returns a **new list** — the original is never mutated:
```python
def classify_segments(wav_path, segments):
    enriched = []
    for seg in segments:
        result = classify_segment(wav_path, seg["start"], seg["end"])
        new_seg = {**seg, **result}  # Shallow copy + merge
        enriched.append(new_seg)
    return enriched  # Original 'segments' list is untouched
```
If the downstream stage crashes, the upstream data is perfectly intact for retry.

---

## 📋 Models Used

| Model | Task | Size | Source |
|-------|------|------|--------|
| [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | Speaker diarization | ~500 MB | pyannote-audio |
| [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) | Emotion classification | ~1.2 GB | HuggingFace |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | Language identification | ~150 MB | OpenAI |

**Emotion labels:** neutral, happy, angry, sad, fear, surprise, disgust

**Language detection:** 99 languages supported (ISO 639-1 codes)

---

## 🚀 Quick Deploy to HuggingFace Spaces

### Prerequisites
1. A [HuggingFace account](https://huggingface.co/join)
2. An access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the pyannote model licenses:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) → "Agree and access"
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) → "Agree and access"

### Deploy in 3 Steps

1. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Space name: `speakerscan`
   - SDK: **Docker** (Select the **Blank** template so it uses our custom `Dockerfile`)
   - Hardware: **CPU Basic** (free — 2 vCPUs, 16 GB RAM)

2. **Add your HF token** in Space Settings → Variables and secrets:
   - Click **New secret**
   - Name: `HF_TOKEN`
   - Value: Your HuggingFace Access Token (with Write access)

3. **Push the code**:
   - Add the Hugging Face Space repository as a Git remote locally and push to it:
     ```bash
     git remote add hf https://huggingface.co/spaces/champTUSHARg007/speakerscan
     git push -f hf main
     ```
   - The Space will automatically build the custom container and run the app. Your app will be live at:
     ```
     https://huggingface.co/spaces/champTUSHARg007/speakerscan
     ```

### Running Locally

```bash
git clone https://github.com/tusharg007/speakerscan.git
cd speakerscan

# Create .env with your token
echo "HF_TOKEN=hf_your_token" > .env

# Install and run
pip install -r requirements.txt
streamlit run app.py
```

### Running with Docker

```bash
docker build -t speakerscan .
docker run -p 8501:8501 -e HF_TOKEN=hf_your_token speakerscan
```

---

## 💻 CLI Usage (Batch Processing)

SpeakerScan also works as a CLI tool for batch processing:

```bash
# Process a list of YouTube URLs
python main.py --input urls.txt --output ./data

# With concurrent workers (I/O overlaps with compute)
python main.py --input urls.txt --output ./data --workers 2

# Process local audio files
python main.py --input local_files.txt --output ./data
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Text file with one URL/path per line |
| `--output` | `./data` | Root output directory |
| `--workers` | `1` | Concurrent file processors |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token (fallback to env var) |

---

## 📁 Output Formats

### JSON Annotations (per-file)

```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.312,
    "end": 3.768,
    "emotion": "neutral",
    "emotion_confidence": 0.847,
    "language": "en",
    "language_confidence": 0.992
  },
  {
    "speaker": "SPEAKER_01",
    "start": 4.012,
    "end": 6.903,
    "emotion": "happy",
    "emotion_confidence": 0.623,
    "language": "hi",
    "language_confidence": 0.988
  }
]
```

### RTTM (NIST Standard Diarization Format)

```
SPEAKER file_id 1 0.312 3.456 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER file_id 1 4.012 2.891 <NA> <NA> SPEAKER_01 <NA> <NA>
```

### CSV Manifest

```csv
file_id,url,status,num_speakers,num_segments,duration_sec,languages_detected,processed_at
abc123,https://youtube.com/...,completed,2,47,180.5,en,2026-06-07T12:00:00+00:00
```

### Checkpoint JSON (for crash recovery)

```json
{
  "abc123": {
    "status": "completed",
    "stage": "annotate",
    "url": "https://youtube.com/...",
    "started_at": "2026-06-07T12:00:00Z",
    "completed_at": "2026-06-07T12:03:45Z",
    "error": null
  }
}
```

---

## 📐 Module Overview

```
speakerscan/
├── app.py                  # Streamlit web frontend
├── ui_helpers.py           # UI utilities — charts, formatting, pipeline threading
├── main.py                 # CLI entry-point & batch orchestrator
├── config.py               # Centralized constants, model IDs, paths
├── checkpoint.py           # JSON-backed idempotency with stage tracking
├── downloader.py           # yt-dlp + ffmpeg audio download & conversion
├── diarizer.py             # pyannote speaker diarization + RTTM writer
├── emotion_classifier.py   # wav2vec2 per-segment emotion classification
├── language_detector.py    # Whisper per-segment language identification
├── annotator.py            # JSON annotation writer + CSV manifest
├── requirements.txt        # Pinned dependencies (CPU-only PyTorch)
├── packages.txt            # System packages for HF Spaces (ffmpeg)
├── Dockerfile              # Docker deployment
├── test_pipeline.py        # E2E integration tests (7 cases)
├── test_streamlit.py       # UI helper unit tests (8 cases)
├── sample_urls.txt         # Public domain test URLs
├── .env.example            # Token setup template
└── demo_outputs/           # Pre-computed demo samples
    └── README.md           # Schema documentation
```

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `app.py` | 444 | Streamlit UI — input tabs, progress tracking, result rendering |
| `ui_helpers.py` | 380 | Plotly timeline, formatting, threaded pipeline runner |
| `main.py` | 442 | CLI orchestrator, batch processing, logging setup |
| `checkpoint.py` | 180 | Thread-safe checkpoint with 5-stage tracking |
| `downloader.py` | 354 | YouTube download, ffmpeg conversion, WAV validation |
| `diarizer.py` | 206 | pyannote integration, RTTM output |
| `emotion_classifier.py` | 206 | wav2vec2 emotion classification (immutable) |
| `language_detector.py` | 188 | Whisper language ID (immutable) |
| `annotator.py` | 215 | JSON/CSV writing with atomic operations |
| `config.py` | 73 | All constants in one place |

---

## 🧪 Testing

### Unit tests (no models required)

```bash
python test_streamlit.py
```
Tests formatting, demo loading, Plotly chart generation, and summary metrics — 8 tests, runs in <1 second.

### Integration test (requires HF_TOKEN + internet)

```bash
python test_pipeline.py
```
Processes a real YouTube URL through all 5 stages and validates:
- WAV format (16kHz, mono)
- RTTM compliance
- JSON schema (all 7 required fields)
- CSV manifest columns
- Checkpoint state
- Idempotency (re-run skips in <2s)

### Idempotency verification

```bash
# First run — downloads, processes, writes outputs
python main.py --input sample_urls.txt --output ./data

# Second run — skips all files instantly (reads checkpoint)
python main.py --input sample_urls.txt --output ./data
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `HF_TOKEN not set` | Create `.env` file or set env var. On HF Spaces: Settings → Secrets |
| `401 Unauthorized` | Accept model licenses at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| `ffmpeg not found` | Install ffmpeg (`apt install ffmpeg` or `pip install imageio-ffmpeg`) |
| `Out of memory` | Use shorter audio (<5 min on free tier). Or upgrade to GPU Space |
| `yt-dlp: video unavailable` | URL may be region-locked, private, or removed |
| Checkpoint stuck `in_progress` | Delete `checkpoint.json` and re-run |

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | Streamlit | 1.45+ |
| Visualization | Plotly | 6.x |
| Diarization | pyannote-audio | 3.3.x |
| Emotion | HuggingFace Transformers | 4.52+ |
| Language ID | OpenAI Whisper | tiny |
| Deep Learning | PyTorch (CPU) | 2.x |
| Audio Processing | librosa + soundfile | 0.11+ |
| Download | yt-dlp + ffmpeg-python | latest |
| Deployment | HuggingFace Spaces | Streamlit SDK |

---

## 📝 License

This project is built for demonstration and portfolio purposes.
Audio content in `sample_urls.txt` is sourced from public domain / Creative Commons sources.
