"""
CLI entry-point and batch orchestrator for the speech annotation pipeline.

Usage::

    python main.py --input urls.txt --output ./data --workers 1

Pipeline per file:
    1. **Checkpoint check** → skip if already completed.
    2. **Download** → yt-dlp + ffmpeg → 16 kHz mono WAV.
    3. **Diarize** → pyannote → segments + RTTM.
    4. **Emotion** → wav2vec2 → enriched segments (immutable copy).
    5. **Language** → Whisper → enriched segments (immutable copy).
    6. **Annotate** → JSON + CSV manifest.

Error isolation: if one file fails at any stage, the error is logged
and the pipeline moves to the next file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env BEFORE importing project modules that read env vars
load_dotenv()

from annotator import update_manifest, write_json_annotations
from checkpoint import CheckpointManager
from config import DEFAULT_WORKERS, ensure_dirs
from diarizer import diarize, write_rttm
from downloader import download_audio, extract_file_id
from emotion_classifier import classify_segments
from language_detector import detect_language_segments


# ── single-file pipeline ───────────────────────────────────────


def process_file(
    url_or_path: str,
    dirs: dict[str, Path],
    checkpoint: CheckpointManager,
    hf_token: str | None = None,
) -> bool:
    """Run the full pipeline for one input URL or local path.

    Args:
        url_or_path: YouTube URL or local file path.
        dirs:        Directory mapping from :func:`config.ensure_dirs`.
        checkpoint:  Shared checkpoint manager.
        hf_token:    Optional HuggingFace token override.

    Returns:
        ``True`` if the file was processed (or already completed);
        ``False`` on failure.
    """
    file_id = extract_file_id(url_or_path)
    pipeline_start = time.perf_counter()

    # ── idempotency ─────────────────────────────────────────
    if checkpoint.is_completed(file_id):
        logger.info("[{}] Already completed — skipping", file_id)
        return True

    checkpoint.mark_started(file_id, url_or_path)

    try:
        # ── Stage 1: Download ───────────────────────────────
        wav_path = None
        if checkpoint.should_run_stage(file_id, "download"):
            logger.info("[{}] ▶ Stage: download", file_id)
            stage_start = time.perf_counter()

            wav_path = download_audio(
                url_or_path,
                raw_dir=dirs["raw"],
                audio_dir=dirs["audio"],
                file_id=file_id,
            )
            if wav_path is None:
                raise RuntimeError("Download failed — see logs above")

            checkpoint.mark_stage(file_id, "download")
            logger.info(
                "[{}] ✓ download ({:.1f}s)",
                file_id,
                time.perf_counter() - stage_start,
            )
        else:
            # Resuming — WAV should already exist
            wav_path = dirs["audio"] / f"{file_id}.wav"
            if not wav_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint says download is done but {wav_path} is missing"
                )
            logger.info("[{}] ⏭ download (already done)", file_id)

        # ── Stage 2: Diarization ────────────────────────────
        segments: list[dict] = []
        if checkpoint.should_run_stage(file_id, "diarize"):
            logger.info("[{}] ▶ Stage: diarize", file_id)
            stage_start = time.perf_counter()

            segments = diarize(wav_path, hf_token=hf_token)
            write_rttm(segments, file_id, dirs["annotations"])

            checkpoint.mark_stage(file_id, "diarize")
            logger.info(
                "[{}] ✓ diarize — {} segments ({:.1f}s)",
                file_id,
                len(segments),
                time.perf_counter() - stage_start,
            )
        else:
            logger.info("[{}] ⏭ diarize (already done)", file_id)
            # Load segments from existing RTTM for downstream stages
            segments = _load_segments_from_rttm(
                dirs["annotations"] / f"{file_id}.rttm", file_id
            )

        # Handle no-speech case
        if not segments:
            logger.warning("[{}] No speech segments — writing empty annotations", file_id)
            write_json_annotations([], file_id, dirs["annotations"])
            update_manifest(
                file_id, url_or_path, "completed", [], wav_path, dirs["annotations"]
            )
            checkpoint.mark_completed(file_id)
            return True

        # ── Stage 3: Emotion classification ─────────────────
        enriched_segments: list[dict] = segments
        if checkpoint.should_run_stage(file_id, "emotion"):
            logger.info("[{}] ▶ Stage: emotion", file_id)
            stage_start = time.perf_counter()

            enriched_segments = classify_segments(wav_path, segments)

            checkpoint.mark_stage(file_id, "emotion")
            logger.info(
                "[{}] ✓ emotion ({:.1f}s)",
                file_id,
                time.perf_counter() - stage_start,
            )
        else:
            logger.info("[{}] ⏭ emotion (already done)", file_id)

        # ── Stage 4: Language detection ─────────────────────
        if checkpoint.should_run_stage(file_id, "language"):
            logger.info("[{}] ▶ Stage: language", file_id)
            stage_start = time.perf_counter()

            enriched_segments = detect_language_segments(wav_path, enriched_segments)

            checkpoint.mark_stage(file_id, "language")
            logger.info(
                "[{}] ✓ language ({:.1f}s)",
                file_id,
                time.perf_counter() - stage_start,
            )
        else:
            logger.info("[{}] ⏭ language (already done)", file_id)

        # ── Stage 5: Write annotations ──────────────────────
        if checkpoint.should_run_stage(file_id, "annotate"):
            logger.info("[{}] ▶ Stage: annotate", file_id)
            stage_start = time.perf_counter()

            write_json_annotations(enriched_segments, file_id, dirs["annotations"])
            update_manifest(
                file_id,
                url_or_path,
                "completed",
                enriched_segments,
                wav_path,
                dirs["annotations"],
            )

            checkpoint.mark_stage(file_id, "annotate")
            logger.info(
                "[{}] ✓ annotate ({:.1f}s)",
                file_id,
                time.perf_counter() - stage_start,
            )
        else:
            logger.info("[{}] ⏭ annotate (already done)", file_id)

        checkpoint.mark_completed(file_id)
        elapsed = time.perf_counter() - pipeline_start
        logger.info("[{}] ════ Pipeline complete ({:.1f}s) ════", file_id, elapsed)
        return True

    except Exception as exc:
        checkpoint.mark_failed(file_id, str(exc))
        update_manifest(
            file_id, url_or_path, "failed", [], None, dirs["annotations"]
        )
        logger.error("[{}] ✗ Pipeline failed: {}", file_id, exc)
        return False


# ── batch orchestration ─────────────────────────────────────────


def run_batch(
    inputs: list[str],
    output_root: Path,
    workers: int = DEFAULT_WORKERS,
    hf_token: str | None = None,
) -> dict[str, int]:
    """Process a batch of URLs/paths through the pipeline.

    Args:
        inputs:      List of YouTube URLs or local file paths.
        output_root: Root output directory.
        workers:     Number of concurrent workers.
        hf_token:    Optional HuggingFace token override.

    Returns:
        Summary dict: ``{"completed": int, "failed": int, "skipped": int}``.
    """
    dirs = ensure_dirs(output_root)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = CheckpointManager(checkpoint_path)

    logger.info(
        "Starting batch: {} files, {} worker(s), output → {}",
        len(inputs),
        workers,
        output_root,
    )

    results = {"completed": 0, "failed": 0, "skipped": 0}

    if workers <= 1:
        # Sequential — simpler debugging, no thread overhead
        for url_or_path in inputs:
            file_id = extract_file_id(url_or_path)
            if checkpoint.is_completed(file_id):
                logger.info("[{}] Already completed — skipping", file_id)
                results["skipped"] += 1
                continue
            success = process_file(url_or_path, dirs, checkpoint, hf_token)
            results["completed" if success else "failed"] += 1
    else:
        # Concurrent — ThreadPoolExecutor with inference locks
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_file, url_or_path, dirs, checkpoint, hf_token
                ): url_or_path
                for url_or_path in inputs
            }
            for future in as_completed(futures):
                url_or_path = futures[future]
                try:
                    success = future.result()
                    results["completed" if success else "failed"] += 1
                except Exception as exc:
                    logger.error("Unhandled error for {}: {}", url_or_path, exc)
                    results["failed"] += 1

    summary = checkpoint.get_summary()
    logger.info(
        "Batch complete — {} completed, {} failed, {} skipped",
        results["completed"],
        results["failed"],
        results["skipped"],
    )
    return results


# ── RTTM loader (for checkpoint resumption) ─────────────────────


def _load_segments_from_rttm(rttm_path: Path, file_id: str) -> list[dict]:
    """Parse an RTTM file back into segment dicts for downstream stages.

    Used when resuming from a checkpoint where diarization already completed.
    """
    segments: list[dict] = []
    if not rttm_path.exists():
        logger.warning("[{}] RTTM not found at {} — cannot resume", file_id, rttm_path)
        return segments

    try:
        for line in rttm_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) >= 9 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append(
                    {
                        "speaker": speaker,
                        "start": round(start, 3),
                        "end": round(start + duration, 3),
                    }
                )
        logger.debug(
            "[{}] Loaded {} segments from RTTM", file_id, len(segments)
        )
    except Exception as exc:
        logger.error("[{}] Failed to parse RTTM: {}", file_id, exc)

    return segments


# ── logging setup ───────────────────────────────────────────────


def _setup_logging(log_dir: Path) -> None:
    """Configure loguru with console + file sinks.

    One log file per run, named with the current timestamp.
    """
    # Remove default stderr handler
    logger.remove()

    # Console — concise, coloured
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{message}</cyan>"
        ),
        colorize=True,
    )

    # File — verbose, structured
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="50 MB",
        encoding="utf-8",
    )

    logger.info("Log file: {}", log_file)


# ── CLI argument parsing ────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        prog="speech-data-pipeline",
        description=(
            "End-to-end speech annotation pipeline: download -> diarize -> "
            "emotion -> language -> annotate."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a text file with one URL or local path per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data"),
        help="Root output directory (default: ./data).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent workers (default: {DEFAULT_WORKERS}). "
        "Inference is serialised via locks; workers overlap I/O with compute.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token. Falls back to HF_TOKEN env var.",
    )
    return parser.parse_args(argv)


# ── main ────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point.  Returns 0 on success, 1 if any file failed."""
    args = parse_args(argv)

    # Resolve output directory and set up logging
    output_root = args.output.resolve()
    dirs = ensure_dirs(output_root)
    _setup_logging(dirs["logs"])

    # Read input file
    input_path = args.input.resolve()
    if not input_path.exists():
        logger.error("Input file not found: {}", input_path)
        return 1

    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    inputs = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    if not inputs:
        logger.warning("No URLs/paths found in {}", input_path)
        return 0

    logger.info("Loaded {} inputs from {}", len(inputs), input_path.name)

    # Resolve HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "No HuggingFace token found.  Set HF_TOKEN env var or use --hf-token."
        )

    # Run the batch
    results = run_batch(inputs, output_root, args.workers, hf_token)

    # Exit code: 0 if no failures, 1 otherwise
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
