"""
Speaker diarization via pyannote-audio.

Wraps ``pyannote/speaker-diarization-3.1`` with:
    * Lazy singleton loading — the model is downloaded and initialized
      once on first use and reused for every subsequent file.
    * Thread-safety — a ``threading.Lock`` guards all inference calls so
      the module is safe to use from a ``ThreadPoolExecutor``.
    * Automatic device selection (CUDA → MPS → CPU).
    * RTTM output writer in standard NIST format.

Environment:
    ``HF_TOKEN`` — A HuggingFace access token with read access to the
    gated pyannote models.  Must be set before calling any function in
    this module.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch
from loguru import logger
from pyannote.audio import Pipeline as PyannotePipeline

from config import DIARIZATION_MODEL

# ── singleton state ─────────────────────────────────────────────
_pipeline: PyannotePipeline | None = None
_pipeline_lock = threading.Lock()

# ── inference lock — serialises diarize() across threads ────────
_inference_lock = threading.Lock()


# ── model loading ───────────────────────────────────────────────


def get_pipeline(hf_token: str | None = None) -> PyannotePipeline:
    """Return the lazily-loaded pyannote diarization pipeline.

    The model is loaded exactly once; subsequent calls return the cached
    instance.  Thread-safe via ``_pipeline_lock``.

    Args:
        hf_token: HuggingFace access token.  Falls back to the
                  ``HF_TOKEN`` environment variable.

    Raises:
        ValueError:  If no token is available.
        RuntimeError: If model download or initialization fails.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        # Double-checked locking
        if _pipeline is not None:
            return _pipeline

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN environment variable not set.  "
                "See README.md for setup instructions."
            )

        device = _select_device()
        logger.info(
            "Loading diarization model {} on {}", DIARIZATION_MODEL, device
        )
        start = time.perf_counter()

        try:
            pipeline = PyannotePipeline.from_pretrained(
                DIARIZATION_MODEL,
                use_auth_token=token,
            )
            pipeline.to(torch.device(device))
            _pipeline = pipeline
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load diarization model.  "
                f"Have you accepted the license at "
                f"https://huggingface.co/{DIARIZATION_MODEL} ?  "
                f"Original error: {exc}"
            ) from exc

        elapsed = time.perf_counter() - start
        logger.info("Diarization model loaded in {:.1f}s", elapsed)
        return _pipeline


# ── diarization ─────────────────────────────────────────────────


def diarize(
    wav_path: Path,
    hf_token: str | None = None,
) -> list[dict]:
    """Run speaker diarization on a 16 kHz mono WAV file.

    Args:
        wav_path:  Path to the audio file.
        hf_token:  Optional HuggingFace token override.

    Returns:
        A list of segment dicts, each containing:
            ``speaker`` (str), ``start`` (float), ``end`` (float).
        Returns an empty list if no speech is detected.
    """
    file_id = wav_path.stem
    logger.info("[{}] Starting diarization", file_id)
    start = time.perf_counter()

    pipeline = get_pipeline(hf_token)

    # Lock ensures only one thread runs inference at a time —
    # pyannote's Pipeline is NOT thread-safe for concurrent forward passes.
    with _inference_lock:
        diarization = pipeline(str(wav_path))

    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
        )

    elapsed = time.perf_counter() - start

    if not segments:
        logger.warning("[{}] No speech segments detected ({:.1f}s)", file_id, elapsed)
    else:
        n_speakers = len({s["speaker"] for s in segments})
        logger.info(
            "[{}] Diarization complete — {} segments, {} speakers ({:.1f}s)",
            file_id,
            len(segments),
            n_speakers,
            elapsed,
        )

    return segments


# ── RTTM output ─────────────────────────────────────────────────


def write_rttm(
    segments: list[dict],
    file_id: str,
    out_dir: Path,
) -> Path:
    """Write diarization segments to an RTTM file (NIST format).

    Format per line::

        SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>

    Args:
        segments: Diarization output from :func:`diarize`.
        file_id:  Identifier used as the RTTM filename and file column.
        out_dir:  Directory for the ``.rttm`` file.

    Returns:
        Path to the written RTTM file.
    """
    rttm_path = out_dir / f"{file_id}.rttm"
    lines: list[str] = []

    for seg in segments:
        duration = round(seg["end"] - seg["start"], 3)
        lines.append(
            f"SPEAKER {file_id} 1 {seg['start']} {duration} "
            f"<NA> <NA> {seg['speaker']} <NA> <NA>"
        )

    # Atomic write
    tmp_path = rttm_path.with_suffix(".rttm.tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(rttm_path)

    logger.debug("[{}] Wrote RTTM with {} entries", file_id, len(lines))
    return rttm_path


# ── helpers ─────────────────────────────────────────────────────


def _select_device() -> str:
    """Pick the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
