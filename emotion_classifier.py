"""
Per-segment emotion classification via wav2vec2.

Uses ``ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`` from
HuggingFace for audio-based emotion recognition.

Design principles:
    * **Lazy singleton** — the model is loaded once and reused.
    * **Thread-safe** — a ``threading.Lock`` guards inference so the
      module works safely from a ``ThreadPoolExecutor``.
    * **Immutable input** — ``classify_segments`` returns a *new* list of
      enriched segment dicts.  The original diarization segments are
      never mutated, so a partial failure leaves them intact.
    * **Graceful degradation** — segments shorter than
      ``MIN_SEGMENT_DURATION`` are tagged ``too_short``; inference errors
      on individual segments are caught and logged.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from loguru import logger
from transformers import pipeline as hf_pipeline

from config import EMOTION_MODEL, MIN_SEGMENT_DURATION, SAMPLE_RATE

# ── singleton state ─────────────────────────────────────────────
_classifier = None
_classifier_lock = threading.Lock()

# ── inference lock — serialises classify calls across threads ───
_inference_lock = threading.Lock()


# ── model loading ───────────────────────────────────────────────


def get_classifier():
    """Return the lazily-loaded emotion classification pipeline.

    Thread-safe via double-checked locking on ``_classifier_lock``.
    """
    global _classifier

    if _classifier is not None:
        return _classifier

    with _classifier_lock:
        if _classifier is not None:
            return _classifier

        device = _select_device()
        logger.info("Loading emotion model {} on {}", EMOTION_MODEL, device)
        start = time.perf_counter()

        _classifier = hf_pipeline(
            "audio-classification",
            model=EMOTION_MODEL,
            device=device,
        )

        elapsed = time.perf_counter() - start
        logger.info("Emotion model loaded in {:.1f}s", elapsed)
        return _classifier


# ── single-segment classification ───────────────────────────────


def classify_segment(
    wav_path: Path,
    start: float,
    end: float,
) -> dict:
    """Classify emotion for a single time-sliced audio segment.

    Args:
        wav_path: Path to the full 16 kHz mono WAV file.
        start:    Segment start time in seconds.
        end:      Segment end time in seconds.

    Returns:
        A dict with ``emotion`` (str) and ``emotion_confidence`` (float).
        Special labels: ``"too_short"`` (segment < 0.5 s),
        ``"error"`` (inference failure).
    """
    duration = end - start

    if duration < MIN_SEGMENT_DURATION:
        return {"emotion": "too_short", "emotion_confidence": 0.0}

    try:
        audio, sr = librosa.load(
            str(wav_path),
            sr=SAMPLE_RATE,
            offset=start,
            duration=duration,
            mono=True,
        )

        # Guard against empty / near-silent segments
        if len(audio) < int(SAMPLE_RATE * MIN_SEGMENT_DURATION):
            return {"emotion": "too_short", "emotion_confidence": 0.0}

        classifier = get_classifier()

        # Thread-safe inference — wav2vec2 is NOT safe for concurrent calls
        with _inference_lock:
            result = classifier(
                {"array": audio.astype(np.float32), "sampling_rate": sr}
            )

        top = result[0]
        return {
            "emotion": top["label"],
            "emotion_confidence": round(top["score"], 3),
        }

    except Exception as exc:
        logger.warning(
            "Emotion classification failed for [{:.1f}–{:.1f}s] in {}: {}",
            start,
            end,
            wav_path.name,
            exc,
        )
        return {"emotion": "error", "emotion_confidence": 0.0}


# ── batch classification ────────────────────────────────────────


def classify_segments(
    wav_path: Path,
    segments: list[dict],
) -> list[dict]:
    """Classify emotion for every segment and return an enriched copy.

    **Does NOT mutate** the input ``segments`` list.  Returns a new list
    where each dict has been shallow-copied and augmented with
    ``emotion`` and ``emotion_confidence`` keys.

    Args:
        wav_path:  Path to the 16 kHz mono WAV.
        segments:  Diarization output — list of ``{speaker, start, end}``.

    Returns:
        A new list of dicts, each containing the original keys plus
        ``emotion`` and ``emotion_confidence``.
    """
    file_id = wav_path.stem
    logger.info("[{}] Classifying emotion for {} segments", file_id, len(segments))
    start_time = time.perf_counter()

    enriched: list[dict] = []
    skipped = 0
    errors = 0

    for i, seg in enumerate(segments, 1):
        result = classify_segment(wav_path, seg["start"], seg["end"])

        # Shallow copy + merge — original dict is untouched
        new_seg = {**seg, **result}
        enriched.append(new_seg)

        if result["emotion"] == "too_short":
            skipped += 1
        elif result["emotion"] == "error":
            errors += 1

        if i % 20 == 0:
            logger.debug("[{}] Emotion progress: {}/{}", file_id, i, len(segments))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "[{}] Emotion classification done — {} ok, {} skipped, {} errors ({:.1f}s)",
        file_id,
        len(enriched) - skipped - errors,
        skipped,
        errors,
        elapsed,
    )

    return enriched


# ── helpers ─────────────────────────────────────────────────────


def _select_device() -> int | str:
    """Pick the best available compute device.

    Returns an int (CUDA device index) or string for the HF pipeline
    ``device`` parameter.
    """
    if torch.cuda.is_available():
        return 0  # HF pipeline expects int for CUDA
    # MPS support in HF pipelines is inconsistent — fall back to CPU
    return -1  # CPU
