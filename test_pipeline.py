"""
End-to-end smoke test for the speech data annotation pipeline.

Processes a single short public-domain audio file through the full
pipeline and validates that all expected output files exist with the
correct structure.

Usage::

    python test_pipeline.py

Requires:
    * ``HF_TOKEN`` environment variable set (or ``.env`` file).
    * ``ffmpeg`` available on PATH.
    * Internet access for downloading the test audio.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from main import process_file, _setup_logging
from checkpoint import CheckpointManager
from config import ensure_dirs

# Short public domain test URL — Universal Declaration of Human Rights (~2 min)
TEST_URL = "https://www.youtube.com/watch?v=5RR4VXNX3jA"
TEST_OUTPUT_DIR = Path(__file__).parent / "_test_output"


class TestPipeline(unittest.TestCase):
    """End-to-end smoke test: process one URL, validate outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create test output directory and set up logging."""
        cls.output_dir = TEST_OUTPUT_DIR.resolve()
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.dirs = ensure_dirs(cls.output_dir)
        _setup_logging(cls.dirs["logs"])

        cls.hf_token = os.environ.get("HF_TOKEN")
        if not cls.hf_token:
            raise unittest.SkipTest("HF_TOKEN not set — skipping integration test")

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test output directory."""
        if TEST_OUTPUT_DIR.exists():
            # Keep outputs for manual inspection; uncomment to auto-clean:
            # shutil.rmtree(TEST_OUTPUT_DIR)
            pass

    def test_01_full_pipeline(self) -> None:
        """Process a single URL through all pipeline stages."""
        checkpoint = CheckpointManager(self.output_dir / "checkpoint.json")

        success = process_file(
            TEST_URL,
            self.dirs,
            checkpoint,
            hf_token=self.hf_token,
        )

        self.assertTrue(success, "Pipeline should complete successfully")

    def test_02_wav_exists(self) -> None:
        """Verify the 16 kHz mono WAV was created."""
        wav_files = list(self.dirs["audio"].glob("*.wav"))
        self.assertGreater(len(wav_files), 0, "At least one WAV should exist")

        # Validate WAV properties
        import soundfile as sf

        for wav in wav_files:
            info = sf.info(str(wav))
            self.assertEqual(info.samplerate, 16000, f"{wav.name}: expected 16kHz")
            self.assertEqual(info.channels, 1, f"{wav.name}: expected mono")

    def test_03_rttm_exists(self) -> None:
        """Verify RTTM file was created with valid format."""
        rttm_files = list(self.dirs["annotations"].glob("*.rttm"))
        self.assertGreater(len(rttm_files), 0, "At least one RTTM should exist")

        for rttm in rttm_files:
            content = rttm.read_text(encoding="utf-8").strip()
            if content:  # May be empty for no-speech audio
                for line in content.splitlines():
                    parts = line.split()
                    self.assertEqual(
                        parts[0], "SPEAKER", f"RTTM line should start with SPEAKER"
                    )
                    self.assertGreaterEqual(
                        len(parts), 9, f"RTTM line should have ≥9 fields"
                    )

    def test_04_json_exists_and_valid(self) -> None:
        """Verify JSON annotation file has correct structure."""
        json_files = list(self.dirs["annotations"].glob("*.json"))
        self.assertGreater(len(json_files), 0, "At least one JSON should exist")

        required_keys = {
            "speaker", "start", "end",
            "emotion", "emotion_confidence",
            "language", "language_confidence",
        }

        for jf in json_files:
            data = json.loads(jf.read_text(encoding="utf-8"))
            self.assertIsInstance(data, list, "JSON should be a list")
            if data:  # May be empty for no-speech audio
                for seg in data:
                    self.assertIsInstance(seg, dict)
                    missing = required_keys - set(seg.keys())
                    self.assertEqual(
                        missing,
                        set(),
                        f"Segment missing keys: {missing}",
                    )
                    self.assertIsInstance(seg["start"], (int, float))
                    self.assertIsInstance(seg["end"], (int, float))
                    self.assertGreater(
                        seg["end"], seg["start"], "end should be > start"
                    )

    def test_05_manifest_exists(self) -> None:
        """Verify manifest CSV was created with at least one row."""
        manifest = self.dirs["annotations"] / "manifest.csv"
        self.assertTrue(manifest.exists(), "manifest.csv should exist")

        import csv

        with open(manifest, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        self.assertGreater(len(rows), 0, "Manifest should have ≥1 row")

        required_cols = {
            "file_id", "url", "status", "num_speakers",
            "num_segments", "duration_sec", "processed_at",
        }
        actual_cols = set(rows[0].keys())
        missing = required_cols - actual_cols
        self.assertEqual(missing, set(), f"Manifest missing columns: {missing}")

    def test_06_checkpoint_state(self) -> None:
        """Verify checkpoint shows completed status."""
        checkpoint = CheckpointManager(self.output_dir / "checkpoint.json")
        state = checkpoint.get_all()

        self.assertGreater(len(state), 0, "Checkpoint should have ≥1 entry")

        for file_id, entry in state.items():
            self.assertEqual(
                entry["status"],
                "completed",
                f"{file_id} should be completed",
            )

    def test_07_idempotency(self) -> None:
        """Verify re-running the same URL skips processing."""
        checkpoint = CheckpointManager(self.output_dir / "checkpoint.json")

        # Second run should be instant (skip)
        import time

        start = time.perf_counter()
        success = process_file(
            TEST_URL,
            self.dirs,
            checkpoint,
            hf_token=self.hf_token,
        )
        elapsed = time.perf_counter() - start

        self.assertTrue(success)
        self.assertLess(
            elapsed, 2.0, "Idempotent re-run should complete in < 2s"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
