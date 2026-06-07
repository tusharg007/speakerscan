"""
Idempotency checkpoint manager.

Tracks per-file processing state across pipeline stages so that a
crashed or interrupted run can be resumed without re-processing
completed work.  State is persisted to a JSON file with atomic writes
(write-to-tmp-then-rename) to prevent corruption on mid-write crashes.

The checkpoint records four pipeline stages in order:
    download → diarize → emotion → language → annotate

Thread-safety is guaranteed via ``threading.Lock`` so the manager can
be shared across a ``ThreadPoolExecutor`` with ``--workers > 1``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# Ordered pipeline stages — used for resumption logic
STAGES: list[str] = ["download", "diarize", "emotion", "language", "annotate"]


class CheckpointManager:
    """JSON-backed, thread-safe checkpoint for per-file pipeline state.

    Parameters:
        checkpoint_path: Absolute path to the JSON checkpoint file.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self._path = checkpoint_path
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    # ── public API ──────────────────────────────────────────────

    def is_completed(self, file_id: str) -> bool:
        """Return ``True`` if *file_id* has already been fully processed."""
        with self._lock:
            entry = self._state.get(file_id, {})
            return entry.get("status") == "completed"

    def get_stage(self, file_id: str) -> str | None:
        """Return the last successfully completed stage for *file_id*,
        or ``None`` if no stage has been recorded yet."""
        with self._lock:
            return self._state.get(file_id, {}).get("stage")

    def should_run_stage(self, file_id: str, stage: str) -> bool:
        """Return ``True`` if *stage* still needs to run for *file_id*.

        A stage should run if:
        - The file has no checkpoint entry, OR
        - The file is not completed and the stage has not been reached yet.
        """
        with self._lock:
            entry = self._state.get(file_id)
            if entry is None:
                return True
            if entry.get("status") == "completed":
                return False
            last_stage = entry.get("stage")
            if last_stage is None:
                return True
            try:
                return STAGES.index(stage) > STAGES.index(last_stage)
            except ValueError:
                return True

    def mark_started(self, file_id: str, url: str) -> None:
        """Record that processing has begun for *file_id*."""
        with self._lock:
            if file_id not in self._state:
                self._state[file_id] = {
                    "status": "in_progress",
                    "stage": None,
                    "url": url,
                    "started_at": _now_iso(),
                    "completed_at": None,
                    "error": None,
                }
            else:
                # Resuming a previously failed file
                self._state[file_id]["status"] = "in_progress"
                self._state[file_id]["error"] = None
            self._save()
        logger.debug("Checkpoint: marked {} as started", file_id)

    def mark_stage(self, file_id: str, stage: str) -> None:
        """Record that *stage* completed successfully for *file_id*."""
        with self._lock:
            if file_id in self._state:
                self._state[file_id]["stage"] = stage
                self._save()
        logger.debug("Checkpoint: {} completed stage '{}'", file_id, stage)

    def mark_completed(self, file_id: str) -> None:
        """Record that *file_id* has been fully processed."""
        with self._lock:
            if file_id in self._state:
                self._state[file_id]["status"] = "completed"
                self._state[file_id]["completed_at"] = _now_iso()
                self._save()
        logger.info("Checkpoint: {} completed successfully", file_id)

    def mark_failed(self, file_id: str, error: str) -> None:
        """Record that *file_id* failed with the given error message."""
        with self._lock:
            if file_id in self._state:
                self._state[file_id]["status"] = "failed"
                self._state[file_id]["error"] = error
                self._save()
        logger.warning("Checkpoint: {} marked as failed — {}", file_id, error)

    def get_summary(self) -> dict[str, int]:
        """Return counts grouped by status (completed / failed / in_progress)."""
        with self._lock:
            summary: dict[str, int] = {"completed": 0, "failed": 0, "in_progress": 0}
            for entry in self._state.values():
                status = entry.get("status", "in_progress")
                summary[status] = summary.get(status, 0) + 1
            return summary

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the full checkpoint state."""
        with self._lock:
            return json.loads(json.dumps(self._state))

    # ── persistence ─────────────────────────────────────────────

    def _load(self) -> None:
        """Load state from disk.  If the file doesn't exist or is corrupt,
        start with an empty state (log a warning on corruption)."""
        if not self._path.exists():
            logger.debug("No checkpoint file found at {} — starting fresh", self._path)
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            self._state = json.loads(raw)
            logger.info(
                "Loaded checkpoint with {} entries from {}",
                len(self._state),
                self._path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupt or unreadable checkpoint at {} — starting fresh ({})",
                self._path,
                exc,
            )
            self._state = {}

    def _save(self) -> None:
        """Atomically persist state: write to a temp file, then rename.

        Must be called while ``self._lock`` is held.
        """
        tmp_path = self._path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(self._state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self._path)
        except OSError as exc:
            logger.error("Failed to save checkpoint: {}", exc)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
