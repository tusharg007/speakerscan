"""
Annotation writer — merges pipeline outputs into structured files.

Produces three output formats:
    1. **Per-file JSON** — full segment-level annotations.
    2. **Per-file RTTM** — standard diarization format (written by
       :mod:`diarizer`, not this module).
    3. **Master CSV manifest** — one row per processed file with summary
       statistics.

All file writes are atomic (write-to-tmp → rename) to prevent corrupt
output on crashes.
"""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from loguru import logger

# Lock for thread-safe manifest CSV appends
_manifest_lock = threading.Lock()


# ── per-file JSON annotations ──────────────────────────────────


def write_json_annotations(
    segments: list[dict],
    file_id: str,
    out_dir: Path,
) -> Path:
    """Write enriched segment annotations to a per-file JSON file.

    Each element in the JSON array contains:
        ``speaker``, ``start``, ``end``, ``emotion``,
        ``emotion_confidence``, ``language``, ``language_confidence``.

    Args:
        segments: Enriched segment list (output of emotion + language stages).
        file_id:  Identifier used for the filename.
        out_dir:  Annotations directory.

    Returns:
        Path to the written JSON file.
    """
    json_path = out_dir / f"{file_id}.json"
    tmp_path = json_path.with_suffix(".json.tmp")

    # Ensure consistent key ordering for readability
    ordered_segments: list[dict] = []
    for seg in segments:
        ordered_segments.append(
            {
                "speaker": seg.get("speaker", "UNKNOWN"),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "emotion": seg.get("emotion", "unknown"),
                "emotion_confidence": seg.get("emotion_confidence", 0.0),
                "language": seg.get("language", "unknown"),
                "language_confidence": seg.get("language_confidence", 0.0),
            }
        )

    tmp_path.write_text(
        json.dumps(ordered_segments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(json_path)

    logger.info("[{}] Wrote JSON annotations ({} segments)", file_id, len(segments))
    return json_path


# ── master CSV manifest ─────────────────────────────────────────

MANIFEST_COLUMNS: list[str] = [
    "file_id",
    "url",
    "status",
    "num_speakers",
    "num_segments",
    "duration_sec",
    "languages_detected",
    "processed_at",
]


def update_manifest(
    file_id: str,
    url: str,
    status: str,
    segments: list[dict],
    wav_path: Path | None,
    out_dir: Path,
) -> Path:
    """Append or update a row in the master CSV manifest.

    Thread-safe via ``_manifest_lock``.

    Args:
        file_id:   Unique file identifier.
        url:       Original source URL or path.
        status:    ``"completed"`` or ``"failed"``.
        segments:  Enriched segment list (may be empty on failure).
        wav_path:  Path to WAV for duration calculation (``None`` on failure).
        out_dir:   Directory containing ``manifest.csv``.

    Returns:
        Path to the manifest CSV.
    """
    manifest_path = out_dir / "manifest.csv"
    summary = get_file_summary(segments, wav_path)

    row = {
        "file_id": file_id,
        "url": url,
        "status": status,
        "num_speakers": summary["num_speakers"],
        "num_segments": summary["num_segments"],
        "duration_sec": summary["duration_sec"],
        "languages_detected": summary["languages_detected"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    with _manifest_lock:
        write_header = not manifest_path.exists()

        with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    logger.debug("[{}] Manifest updated (status={})", file_id, status)
    return manifest_path


# ── summary helpers ─────────────────────────────────────────────


def get_file_summary(
    segments: list[dict],
    wav_path: Path | None,
) -> dict:
    """Compute summary statistics for one processed file.

    Args:
        segments: Enriched segment list.
        wav_path: WAV file path for duration (can be ``None``).

    Returns:
        Dict with ``num_speakers``, ``num_segments``, ``duration_sec``,
        ``languages_detected``.
    """
    num_speakers = len({s.get("speaker") for s in segments}) if segments else 0
    num_segments = len(segments)

    # Duration from the WAV file itself (more reliable than segment spans)
    duration_sec = 0.0
    if wav_path and wav_path.exists():
        try:
            info = sf.info(str(wav_path))
            duration_sec = round(info.frames / info.samplerate, 2)
        except Exception:
            # Fallback: use segment timestamps
            if segments:
                duration_sec = round(max(s.get("end", 0) for s in segments), 2)

    # Unique languages detected (excluding error / too_short)
    languages = sorted(
        {
            s.get("language", "unknown")
            for s in segments
            if s.get("language") not in ("too_short", "error", "unknown", None)
        }
    )
    languages_str = ",".join(languages) if languages else "unknown"

    return {
        "num_speakers": num_speakers,
        "num_segments": num_segments,
        "duration_sec": duration_sec,
        "languages_detected": languages_str,
    }


def get_output_as_dict(json_path: Path) -> list[dict]:
    """Read a previously written JSON annotation file back into memory.

    Args:
        json_path: Path to the ``.json`` annotation file.

    Returns:
        List of segment dicts, or empty list if file is missing/corrupt.
    """
    if not json_path.exists():
        logger.warning("Annotation file not found: {}", json_path)
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        logger.warning("Unexpected JSON structure in {}", json_path)
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read annotation {}: {}", json_path, exc)
        return []

