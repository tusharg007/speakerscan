"""
UI helper functions for the Streamlit app.

Extracted to keep app.py under 600 lines.  Contains formatting,
chart generation, demo sample loading, and pipeline runner logic.
"""

from __future__ import annotations

import json
import logging
import queue
import tempfile
import threading
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger("speech_pipeline")

# ── Constants ───────────────────────────────────────────────────

EMOTION_COLORS: dict[str, str] = {
    "neutral": "#9E9E9E",
    "happy": "#4CAF50",
    "angry": "#F44336",
    "sad": "#2196F3",
    "fear": "#FF9800",
    "surprise": "#9C27B0",
    "disgust": "#795548",
    "too_short": "#BDBDBD",
    "error": "#E0E0E0",
    "unknown": "#BDBDBD",
}

STAGE_WEIGHTS: dict[str, int] = {
    "download": 20,
    "diarize": 45,
    "emotion": 65,
    "language": 80,
    "annotate": 100,
}

LANGUAGE_OPTIONS: list[str] = [
    "Auto-detect",
    "Hindi",
    "English",
    "Tamil",
    "Telugu",
    "Kannada",
    "Bengali",
    "Marathi",
]

SUPPORTED_FORMATS: list[str] = ["wav", "mp3", "m4a", "ogg"]

DEMO_SAMPLES: list[dict[str, str]] = [
    {"label": "Sample 1 -- Hindi conversation (45s)", "file": "demo_1.json"},
    {"label": "Sample 2 -- English monologue (30s)", "file": "demo_2.json"},
    {"label": "Sample 3 -- Code-switch Hindi-English (60s)", "file": "demo_3.json"},
]


# ── Formatting ──────────────────────────────────────────────────


def format_time(seconds: float) -> str:
    """Format seconds into human-readable mm:ss or h:mm:ss."""
    if seconds < 0:
        return "0:00"
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_confidence(value: float) -> str:
    """Format a 0-1 confidence score as a percentage string."""
    return f"{value * 100:.1f}%"


# ── Demo sample loading ────────────────────────────────────────


def load_demo_sample(demo_dir: Path, filename: str) -> list[dict] | None:
    """Load a pre-computed demo JSON file, or return None if missing."""
    path = demo_dir / filename
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load demo %s: %s", filename, exc)
    return None


# ── Result computation ──────────────────────────────────────────


def compute_summary_metrics(segments: list[dict], duration_sec: float) -> dict[str, Any]:
    """Compute the four summary metric values from enriched segments."""
    if not segments:
        return {
            "duration": "0:00",
            "num_speakers": 0,
            "dominant_language": "N/A",
            "dominant_emotion": "N/A",
        }

    speakers = {s.get("speaker") for s in segments}
    emotions = [
        s.get("emotion", "unknown")
        for s in segments
        if s.get("emotion") not in ("too_short", "error", None)
    ]
    languages = [
        s.get("language", "unknown")
        for s in segments
        if s.get("language") not in ("too_short", "error", None)
    ]

    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "N/A"
    dominant_language = Counter(languages).most_common(1)[0][0] if languages else "N/A"

    # Use segment timestamps as duration fallback
    if duration_sec <= 0 and segments:
        duration_sec = max(s.get("end", 0) for s in segments)

    return {
        "duration": format_time(duration_sec),
        "num_speakers": len(speakers),
        "dominant_language": dominant_language,
        "dominant_emotion": dominant_emotion,
    }


# ── Plotly timeline chart ───────────────────────────────────────


def build_speaker_timeline(segments: list[dict]) -> go.Figure | None:
    """Build a Plotly Gantt chart of speaker segments colored by emotion."""
    if not segments:
        return None

    # Convert to format px.timeline expects (datetime strings for x-axis)
    # We'll use a bar chart instead for seconds-based x-axis
    rows = []
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        emotion = seg.get("emotion", "unknown")
        rows.append({
            "Speaker": seg.get("speaker", "UNKNOWN"),
            "Start": start,
            "End": end,
            "Duration": round(end - start, 3),
            "Emotion": emotion,
            "Language": seg.get("language", "?"),
            "Emotion_Conf": format_confidence(seg.get("emotion_confidence", 0)),
            "Lang_Conf": format_confidence(seg.get("language_confidence", 0)),
            "Color": EMOTION_COLORS.get(emotion, "#9E9E9E"),
        })

    df = pd.DataFrame(rows)

    fig = px.timeline(
        df,
        x_start=pd.to_datetime(df["Start"], unit="s", origin="unix"),
        x_end=pd.to_datetime(df["End"], unit="s", origin="unix"),
        y="Speaker",
        color="Emotion",
        color_discrete_map=EMOTION_COLORS,
        custom_data=["Start", "End", "Emotion", "Language", "Emotion_Conf", "Lang_Conf"],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Time: %{customdata[0]:.1f}s - %{customdata[1]:.1f}s<br>"
            "Emotion: %{customdata[2]} (%{customdata[4]})<br>"
            "Language: %{customdata[3]} (%{customdata[5]})<br>"
            "<extra></extra>"
        )
    )

    # Convert x-axis back to seconds display
    max_time = max(s.get("end", 0) for s in segments) if segments else 60
    tick_count = min(int(max_time / 10) + 1, 20)
    tick_vals = [pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=i * (max_time / tick_count))
                 for i in range(tick_count + 1)]
    tick_text = [f"{i * (max_time / tick_count):.0f}s" for i in range(tick_count + 1)]

    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_text,
        title_text="Time (seconds)",
    )
    fig.update_yaxes(title_text="Speaker", autorange="reversed")
    fig.update_layout(
        height=max(250, len(set(df["Speaker"])) * 80 + 100),
        margin=dict(l=20, r=20, t=40, b=40),
        legend_title_text="Emotion",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ── Segment table ───────────────────────────────────────────────


def segments_to_dataframe(segments: list[dict]) -> pd.DataFrame:
    """Convert enriched segments to a display-ready DataFrame."""
    rows = []
    for seg in segments:
        rows.append({
            "Start": format_time(seg.get("start", 0)),
            "End": format_time(seg.get("end", 0)),
            "Speaker": seg.get("speaker", "UNKNOWN"),
            "Duration(s)": round(seg.get("end", 0) - seg.get("start", 0), 2),
            "Emotion": seg.get("emotion", "unknown"),
            "Confidence%": format_confidence(seg.get("emotion_confidence", 0)),
            "Language": seg.get("language", "unknown"),
            "Lang_confidence%": format_confidence(seg.get("language_confidence", 0)),
        })
    return pd.DataFrame(rows)


def segments_to_csv_bytes(segments: list[dict]) -> bytes:
    """Convert segments to CSV bytes for download."""
    df = segments_to_dataframe(segments)
    return df.to_csv(index=False).encode("utf-8")


# ── Pipeline runner (threaded) ──────────────────────────────────


class PipelineProgress:
    """Thread-safe progress tracker using a queue."""

    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()
        self.error: str | None = None
        self.result: list[dict] | None = None
        self.duration_sec: float = 0.0
        self.done: bool = False

    def update(self, stage: str, message: str, elapsed: float = 0.0) -> None:
        """Push a progress update to the queue."""
        self.queue.put({
            "stage": stage,
            "message": message,
            "progress": STAGE_WEIGHTS.get(stage, 0) / 100.0,
            "elapsed": elapsed,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })


def run_pipeline_thread(
    wav_path: Path,
    output_dir: Path,
    file_id: str,
    hf_token: str | None,
    progress: PipelineProgress,
) -> None:
    """Run the full pipeline in a background thread, posting progress updates."""
    import os
    os.environ.setdefault("HF_TOKEN", hf_token or "")

    try:
        from config import ensure_dirs
        from diarizer import diarize, write_rttm
        from emotion_classifier import classify_segments
        from language_detector import detect_language_segments
        from annotator import write_json_annotations

        dirs = ensure_dirs(output_dir)

        # Stage 1: Diarization
        progress.update("diarize", "Running speaker diarization...")
        stage_start = time.perf_counter()
        segments = diarize(wav_path, hf_token=hf_token)
        diarize_time = time.perf_counter() - stage_start
        progress.update("diarize", f"Diarization complete: {len(segments)} segments", diarize_time)

        if not segments:
            progress.result = []
            progress.done = True
            progress.update("annotate", "No speech detected", 0)
            return

        write_rttm(segments, file_id, dirs["annotations"])

        # Stage 2: Emotion classification
        progress.update("emotion", "Classifying emotions...")
        stage_start = time.perf_counter()
        enriched = classify_segments(wav_path, segments)
        emotion_time = time.perf_counter() - stage_start
        progress.update("emotion", f"Emotion classification complete", emotion_time)

        # Stage 3: Language detection
        progress.update("language", "Detecting languages...")
        stage_start = time.perf_counter()
        enriched = detect_language_segments(wav_path, enriched)
        lang_time = time.perf_counter() - stage_start
        progress.update("language", f"Language detection complete", lang_time)

        # Stage 4: Write annotations
        progress.update("annotate", "Writing annotations...")
        stage_start = time.perf_counter()
        write_json_annotations(enriched, file_id, dirs["annotations"])
        annot_time = time.perf_counter() - stage_start
        progress.update("annotate", "Annotations saved", annot_time)

        # Get duration from WAV
        try:
            import soundfile as sf
            info = sf.info(str(wav_path))
            progress.duration_sec = info.frames / info.samplerate
        except Exception:
            if enriched:
                progress.duration_sec = max(s.get("end", 0) for s in enriched)

        progress.result = enriched
        progress.done = True

    except Exception as exc:
        logger.error("Pipeline thread error: %s", exc)
        progress.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        progress.done = True


def convert_uploaded_file(uploaded_file, output_dir: Path) -> tuple[Path | None, str]:
    """Save uploaded file and convert to 16kHz mono WAV."""
    from downloader import _convert_to_wav, _validate_wav

    file_id = Path(uploaded_file.name).stem.replace(" ", "_")[:32]
    raw_path = output_dir / f"{file_id}_raw{Path(uploaded_file.name).suffix}"
    wav_path = output_dir / f"{file_id}.wav"

    # Save uploaded bytes
    raw_path.write_bytes(uploaded_file.getvalue())

    try:
        _convert_to_wav(raw_path, wav_path)
        if not _validate_wav(wav_path):
            return None, file_id
        return wav_path, file_id
    except Exception as exc:
        logger.error("Conversion failed: %s", exc)
        return None, file_id
    finally:
        if raw_path.exists():
            raw_path.unlink(missing_ok=True)


def download_youtube_audio(url: str, output_dir: Path) -> tuple[Path | None, str]:
    """Download YouTube audio and convert to WAV."""
    from downloader import download_audio, extract_file_id

    file_id = extract_file_id(url)
    raw_dir = output_dir / "raw"
    audio_dir = output_dir / "audio"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    wav_path = download_audio(url, raw_dir, audio_dir, file_id=file_id)
    return wav_path, file_id
