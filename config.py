"""
Centralized configuration for the speech data annotation pipeline.

All magic strings, model identifiers, numeric constants, and directory
conventions live here.  Import from this module instead of scattering
literals across the codebase.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Audio processing
# ──────────────────────────────────────────────
SAMPLE_RATE: int = 16_000          # 16 kHz — standard for speech models
MONO_CHANNELS: int = 1             # mono audio required by all models
MIN_AUDIO_DURATION: float = 1.0    # skip files shorter than 1 s
MIN_SEGMENT_DURATION: float = 0.5  # skip emotion/lang for segments < 0.5 s

# ──────────────────────────────────────────────
# Retry / resilience
# ──────────────────────────────────────────────
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 2.0    # exponential back-off: 2^attempt seconds

# ──────────────────────────────────────────────
# Model identifiers (HuggingFace Hub)
# ──────────────────────────────────────────────
DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
EMOTION_MODEL: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
LANGUAGE_MODEL: str = "openai/whisper-tiny"  # lightweight, good enough for lang-ID

# ──────────────────────────────────────────────
# Directory layout  (relative to --output root)
# ──────────────────────────────────────────────
RAW_DIR: str = "raw"
AUDIO_DIR: str = "audio"
ANNOTATIONS_DIR: str = "annotations"
LOGS_DIR: str = "logs"

# ──────────────────────────────────────────────
# File names
# ──────────────────────────────────────────────
CHECKPOINT_FILE: str = "checkpoint.json"
MANIFEST_FILE: str = "manifest.csv"

# ──────────────────────────────────────────────
# Workers
# ──────────────────────────────────────────────
DEFAULT_WORKERS: int = 1  # safe default; bump with --workers

# ──────────────────────────────────────────────
# Deployment mode
# ──────────────────────────────────────────────
DEMO_MODE: bool = False  # Set True to enable demo sample buttons in Streamlit UI


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    """Create the standard directory tree under *output_root* and return a
    mapping of logical names → resolved ``Path`` objects.

    Returns:
        dict with keys ``"raw"``, ``"audio"``, ``"annotations"``, ``"logs"``.
    """
    dirs: dict[str, Path] = {
        "raw": output_root / RAW_DIR,
        "audio": output_root / AUDIO_DIR,
        "annotations": output_root / ANNOTATIONS_DIR,
        "logs": output_root / LOGS_DIR,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
