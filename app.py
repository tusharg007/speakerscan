"""
Streamlit frontend for the Speech Data Annotation Pipeline.

Provides a web UI for uploading audio or pasting YouTube URLs,
running the full diarization + emotion + language pipeline,
and visualising results with interactive Plotly charts.

Designed for deployment on HuggingFace Spaces (CPU Basic, 16 GB RAM).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Logging (no print statements) ───────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("speech_pipeline")

# ── Page config (must be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="SpeakerScan",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────

APP_TITLE = "SpeakerScan"
APP_SUBTITLE = "Speaker diarization · Emotion · Language ID"
GITHUB_LINK = "https://github.com/tusharg007/speakerscan"
MAX_UPLOAD_MB = 200
DEMO_DIR = Path(__file__).parent / "demo_outputs"

MODEL_LINKS = {
    "pyannote/speaker-diarization-3.1": "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "wav2vec2 emotion": "https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny",
}

# Import helpers (after page config)
from ui_helpers import (
    DEMO_SAMPLES,
    LANGUAGE_OPTIONS,
    SUPPORTED_FORMATS,
    STAGE_WEIGHTS,
    PipelineProgress,
    build_speaker_timeline,
    compute_summary_metrics,
    convert_uploaded_file,
    download_youtube_audio,
    format_time,
    load_demo_sample,
    run_pipeline_thread,
    segments_to_csv_bytes,
    segments_to_dataframe,
)

# ── Sidebar ─────────────────────────────────────────────────────


def render_sidebar() -> None:
    """Render the sidebar with about section and model links."""
    with st.sidebar:
        st.title("🎙️ About")
        st.markdown(
            "This pipeline processes audio through **5 stages**:\n\n"
            "1. **Download** — fetch audio from YouTube or upload\n"
            "2. **Diarize** — identify who speaks when (pyannote)\n"
            "3. **Emotion** — classify emotion per segment (wav2vec2)\n"
            "4. **Language** — detect language per segment (Whisper)\n"
            "5. **Annotate** — produce structured JSON/CSV output"
        )

        st.divider()
        st.subheader("🔗 Links")
        st.markdown(f"[GitHub Repository]({GITHUB_LINK})")

        st.divider()
        st.subheader("🤖 Models Used")
        for name, url in MODEL_LINKS.items():
            st.markdown(f"- [{name}]({url})")

        st.divider()
        # Token status
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            st.success("HF_TOKEN: configured", icon="✅")
        else:
            st.warning(
                "HF_TOKEN not set. Add it in Settings > Secrets on HF Spaces, "
                "or create a .env file locally.",
                icon="⚠️",
            )

        st.caption("Built for Sarvam AI · TTS/Dubbing Pipeline")


# ── Token check ─────────────────────────────────────────────────


def check_hf_token() -> str | None:
    """Check for HF token and show warning if missing."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        st.warning(
            "⚠️ **HuggingFace token not found.**\n\n"
            "The diarization model requires authentication.\n\n"
            "**On HF Spaces:** Go to Settings → Secrets → Add `HF_TOKEN`\n\n"
            "**Locally:** Create a `.env` file with `HF_TOKEN=hf_...`",
            icon="🔐",
        )
        return None
    return token


# ── Demo samples section ────────────────────────────────────────


def render_demo_samples() -> list[dict] | None:
    """Render demo sample buttons. Returns segments if a demo was loaded."""
    st.divider()
    st.markdown("**Or try a demo sample:**")
    cols = st.columns(len(DEMO_SAMPLES))

    for i, sample in enumerate(DEMO_SAMPLES):
        with cols[i]:
            if st.button(sample["label"], key=f"demo_{i}", use_container_width=True):
                data = load_demo_sample(DEMO_DIR, sample["file"])
                if data is not None:
                    return data
                st.info(
                    "Demo sample not yet populated -- run the pipeline first "
                    "to generate it.",
                    icon="ℹ️",
                )
                return None
    return None


# ── Processing section ──────────────────────────────────────────


def run_and_show_progress(
    wav_path: Path,
    output_dir: Path,
    file_id: str,
    hf_token: str | None,
) -> list[dict] | None:
    """Run pipeline in background thread and show live progress."""
    import threading

    progress = PipelineProgress()

    # Start pipeline thread
    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(wav_path, output_dir, file_id, hf_token, progress),
        daemon=True,
    )
    thread.start()

    # Progress display
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_container = st.status("Pipeline running...", expanded=True)
    log_lines: list[str] = []

    while not progress.done:
        try:
            update = progress.queue.get(timeout=0.5)
            pct = update["progress"]
            msg = update["message"]
            elapsed = update["elapsed"]
            ts = update["timestamp"]

            progress_bar.progress(pct, text=msg)
            elapsed_str = f" ({elapsed:.1f}s)" if elapsed > 0 else ""
            log_line = f"`{ts}` {msg}{elapsed_str}"
            log_lines.append(log_line)

            with status_container:
                for line in log_lines:
                    st.markdown(line)

        except Exception:
            pass  # queue.Empty — just poll again

    # Final state
    thread.join(timeout=5)

    if progress.error:
        progress_bar.progress(1.0, text="Pipeline failed")
        status_container.update(label="Pipeline failed", state="error")
        _handle_pipeline_error(progress.error)
        return None

    progress_bar.progress(1.0, text="Pipeline complete!")
    status_container.update(label="Pipeline complete", state="complete")
    return progress.result


# ── Error handling ──────────────────────────────────────────────


def _handle_pipeline_error(error_text: str) -> None:
    """Display appropriate error message based on error type."""
    if "out of memory" in error_text.lower():
        st.error(
            "💾 **Out of Memory Error**\n\n"
            "The audio file is too long for the available RAM (16 GB). "
            "Try a shorter audio clip (< 5 minutes recommended on free tier).",
            icon="🚨",
        )
    elif "hf_token" in error_text.lower() or "401" in error_text:
        st.warning(
            "🔐 **Authentication Error**\n\n"
            "The HuggingFace token is missing or invalid. "
            "Add `HF_TOKEN` in Settings > Secrets.",
            icon="⚠️",
        )
    elif "ffmpeg" in error_text.lower():
        st.error(
            "🔧 **FFmpeg Error**\n\n"
            "FFmpeg is not available. If running on HF Spaces, ensure "
            "`packages.txt` contains `ffmpeg`. Locally, install ffmpeg "
            "and add it to PATH.",
            icon="🚨",
        )
    else:
        st.error("Pipeline failed. See details below.", icon="🚨")

    with st.expander("Full error traceback", expanded=False):
        st.code(error_text, language="text")


# ── Results section ─────────────────────────────────────────────


def render_results(segments: list[dict], duration_sec: float = 0.0) -> None:
    """Render the full results section: metrics, timeline, table, downloads."""
    if not segments:
        st.info("No speech segments were detected in this audio.", icon="🔇")
        return

    st.header("📊 Results")

    # A. Summary metrics
    metrics = compute_summary_metrics(segments, duration_sec)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", metrics["duration"])
    col2.metric("Speakers", metrics["num_speakers"])
    col3.metric("Dominant Language", metrics["dominant_language"])
    col4.metric("Dominant Emotion", metrics["dominant_emotion"])

    st.divider()

    # B. Speaker timeline
    st.subheader("🎤 Speaker Timeline")
    fig = build_speaker_timeline(segments)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No segments to display.")

    st.divider()

    # C. Segment table
    st.subheader("📋 Segment Details")
    df = segments_to_dataframe(segments)
    st.dataframe(df, use_container_width=True, height=400)

    csv_bytes = segments_to_csv_bytes(segments)
    st.download_button(
        label="📥 Export table as CSV",
        data=csv_bytes,
        file_name="segments.csv",
        mime="text/csv",
    )

    st.divider()

    # D. Download section
    st.subheader("💾 Download Annotations")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        json_str = json.dumps(segments, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON",
            data=json_str.encode("utf-8"),
            file_name="annotations.json",
            mime="application/json",
            use_container_width=True,
        )

    with dl_col2:
        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name="annotations.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ── Main app ────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit app entry point."""
    # Load .env for local development
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    render_sidebar()

    # Header
    st.title(f"🎙️ {APP_TITLE}")
    st.caption(APP_SUBTITLE)

    hf_token = check_hf_token()

    # ── Input section ───────────────────────────────────────
    st.header("📂 Input")

    tab_upload, tab_youtube = st.tabs(["📁 Upload audio", "🔗 YouTube URL"])

    uploaded_file = None
    youtube_url = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=SUPPORTED_FORMATS,
            help=f"Supported: {', '.join(SUPPORTED_FORMATS)}. Max {MAX_UPLOAD_MB} MB.",
        )

    with tab_youtube:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        st.selectbox(
            "Audio quality",
            options=["128k", "192k", "320k"],
            index=0,
            key="audio_quality",
        )

    # Controls below both tabs
    col_lang, col_dur = st.columns(2)
    with col_lang:
        st.selectbox("Expected language", LANGUAGE_OPTIONS, key="expected_language")
    with col_dur:
        st.slider(
            "Min segment duration (s)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            key="min_segment_duration",
        )

    # Demo samples
    demo_result = render_demo_samples()
    if demo_result is not None:
        st.session_state["pipeline_result"] = demo_result
        st.session_state["pipeline_duration"] = max(s.get("end", 0) for s in demo_result) if demo_result else 0

    st.divider()

    # Run button
    run_disabled = (not hf_token) and (uploaded_file is None and not youtube_url)
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True, disabled=not hf_token):
        if not uploaded_file and not youtube_url:
            st.warning("Please upload an audio file or enter a YouTube URL.")
            return

        # Create temp directory for this run
        work_dir = Path(tempfile.mkdtemp(prefix="speech_pipeline_"))
        logger.info("Work directory: %s", work_dir)

        try:
            with st.spinner("Preparing audio..."):
                if uploaded_file:
                    wav_path, file_id = convert_uploaded_file(uploaded_file, work_dir)
                    if wav_path is None:
                        st.error("Failed to convert uploaded audio to WAV.")
                        return
                    st.success(f"Audio converted: {file_id}.wav")
                else:
                    # Download stage with progress
                    st.info("Downloading audio from YouTube...")
                    wav_path, file_id = download_youtube_audio(youtube_url, work_dir)
                    if wav_path is None:
                        st.error(
                            "Failed to download audio. Check the URL and try again. "
                            "YouTube may be rate-limiting or the video may be unavailable."
                        )
                        return
                    st.success(f"Audio downloaded: {file_id}.wav")

            # Run pipeline
            result = run_and_show_progress(wav_path, work_dir, file_id, hf_token)

            if result is not None:
                st.session_state["pipeline_result"] = result
                # Get duration
                try:
                    import soundfile as sf
                    info = sf.info(str(wav_path))
                    st.session_state["pipeline_duration"] = info.frames / info.samplerate
                except Exception:
                    if result:
                        st.session_state["pipeline_duration"] = max(
                            s.get("end", 0) for s in result
                        )
                    else:
                        st.session_state["pipeline_duration"] = 0

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}\n\n{__import__('traceback').format_exc()}"
            _handle_pipeline_error(error_text)

    # ── Results section ─────────────────────────────────────
    if "pipeline_result" in st.session_state:
        render_results(
            st.session_state["pipeline_result"],
            st.session_state.get("pipeline_duration", 0),
        )


if __name__ == "__main__":
    main()
