"""
Per-segment language identification via Whisper.

Uses ``openai/whisper-tiny`` for lightweight language detection on each
diarized segment.  This is especially relevant for Sarvam AI's
multilingual TTS work — knowing the language per segment enables
language-specific synthesis and quality filtering.

Design principles mirror :mod:`emotion_classifier`:
    * Lazy singleton model loading.
    * Thread-safe inference via ``threading.Lock``.
    * Immutable input — returns a new enriched list.
    * Graceful degradation for short or problematic segments.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import whisper
from loguru import logger

from config import LANGUAGE_MODEL, MIN_SEGMENT_DURATION, SAMPLE_RATE

# ── singleton state ─────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()

# ── inference lock ──────────────────────────────────────────────
_inference_lock = threading.Lock()


# ── model loading ───────────────────────────────────────────────


def get_model():
    """Return the lazily-loaded Whisper model for language detection.

    Uses the ``tiny`` variant — small enough to co-exist in memory with
    pyannote and wav2vec2, yet accurate enough for language ID (which
    only looks at the first 30 s of audio).
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Extract the model size from the config (e.g. "openai/whisper-tiny" → "tiny")
        model_size = LANGUAGE_MODEL.split("-")[-1] if "/" in LANGUAGE_MODEL else LANGUAGE_MODEL
        logger.info("Loading Whisper ({}) for language detection on {}", model_size, device)
        start = time.perf_counter()

        _model = whisper.load_model(model_size, device=device)

        elapsed = time.perf_counter() - start
        logger.info("Whisper model loaded in {:.1f}s", elapsed)
        return _model


# ── single-segment language detection ───────────────────────────


def detect_language_segment(
    wav_path: Path,
    start: float,
    end: float,
) -> dict:
    """Detect the spoken language for a single time-sliced segment.

    Args:
        wav_path: Path to the 16 kHz mono WAV.
        start:    Segment start in seconds.
        end:      Segment end in seconds.

    Returns:
        ``{"language": str, "language_confidence": float}``
        Special values: ``"too_short"`` or ``"error"`` for the language field.
    """
    duration = end - start

    if duration < MIN_SEGMENT_DURATION:
        return {"language": "too_short", "language_confidence": 0.0}

    try:
        audio, _ = librosa.load(
            str(wav_path),
            sr=SAMPLE_RATE,
            offset=start,
            duration=duration,
            mono=True,
        )

        if len(audio) < int(SAMPLE_RATE * MIN_SEGMENT_DURATION):
            return {"language": "too_short", "language_confidence": 0.0}

        # Whisper expects float32 numpy array
        audio = audio.astype(np.float32)

        # Pad or trim to 30 s (Whisper's expected input length for lang detection)
        audio = whisper.pad_or_trim(audio)

        model = get_model()

        with _inference_lock:
            # Compute log-mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            # Detect language
            _, probs = model.detect_language(mel)

        # Get top language
        top_lang = max(probs, key=probs.get)
        top_conf = round(probs[top_lang], 3)

        return {"language": top_lang, "language_confidence": top_conf}

    except Exception as exc:
        logger.warning(
            "Language detection failed for [{:.1f}–{:.1f}s] in {}: {}",
            start,
            end,
            wav_path.name,
            exc,
        )
        return {"language": "error", "language_confidence": 0.0}


# ── batch detection ─────────────────────────────────────────────


def detect_language_segments(
    wav_path: Path,
    segments: list[dict],
) -> list[dict]:
    """Detect language for every segment and return an enriched copy.

    **Does NOT mutate** the input list.

    Args:
        wav_path:  Path to the 16 kHz mono WAV.
        segments:  List of segment dicts (must have ``start`` and ``end``).

    Returns:
        New list with ``language`` and ``language_confidence`` keys added.
    """
    file_id = wav_path.stem
    logger.info("[{}] Detecting language for {} segments", file_id, len(segments))
    start_time = time.perf_counter()

    enriched: list[dict] = []
    skipped = 0
    errors = 0

    for i, seg in enumerate(segments, 1):
        result = detect_language_segment(wav_path, seg["start"], seg["end"])

        new_seg = {**seg, **result}
        enriched.append(new_seg)

        if result["language"] == "too_short":
            skipped += 1
        elif result["language"] == "error":
            errors += 1

        if i % 20 == 0:
            logger.debug("[{}] Language progress: {}/{}", file_id, i, len(segments))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "[{}] Language detection done — {} ok, {} skipped, {} errors ({:.1f}s)",
        file_id,
        len(enriched) - skipped - errors,
        skipped,
        errors,
        elapsed,
    )

    return enriched
