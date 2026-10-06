"""Offline regression tests for CLI checkpoint recovery."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from checkpoint import CheckpointManager
from config import ensure_dirs


class ResumeTests(unittest.TestCase):
    def test_resume_recomputes_enrichment_and_recovers_bad_rttm(self):
        for rttm_contents in (
            "SPEAKER clip 1 0.0 2.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n",
            "corrupt RTTM\n",
            None,
        ):
            with self.subTest(rttm_contents=rttm_contents), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dirs = ensure_dirs(root)
                (dirs["audio"] / "clip.wav").write_bytes(b"test fixture")
                if rttm_contents is not None:
                    (dirs["annotations"] / "clip.rttm").write_text(rttm_contents, encoding="utf-8")

                checkpoint = CheckpointManager(root / "checkpoint.json")
                checkpoint.mark_started("clip", "clip.wav")
                for stage in ("download", "diarize", "emotion", "language"):
                    checkpoint.mark_stage("clip", stage)

                turns = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}]
                with (
                    patch.object(main, "_validate_wav", return_value=True),
                    patch.object(main, "diarize", return_value=turns) as diarize,
                    patch.object(main, "classify_segments", side_effect=lambda wav, segs: [
                        {**s, "emotion": "neutral", "emotion_confidence": 0.9} for s in segs
                    ]) as emotion,
                    patch.object(main, "detect_language_segments", side_effect=lambda wav, segs: [
                        {**s, "language": "en", "language_confidence": 0.8} for s in segs
                    ]) as language,
                    patch.object(main, "update_manifest"),
                ):
                    self.assertTrue(main.process_file("clip.wav", dirs, checkpoint))

                if rttm_contents is not None and rttm_contents.startswith("SPEAKER"):
                    diarize.assert_not_called()
                else:
                    diarize.assert_called_once()
                emotion.assert_called_once()
                language.assert_called_once()
                data = json.loads((dirs["annotations"] / "clip.json").read_text(encoding="utf-8"))
                self.assertEqual(data[0]["emotion"], "neutral")
                self.assertEqual(data[0]["language"], "en")
                self.assertTrue(checkpoint.is_completed("clip"))


if __name__ == "__main__":
    unittest.main()
