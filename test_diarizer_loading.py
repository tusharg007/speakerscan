"""Regression coverage for dependency errors versus model-access errors."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import diarizer


class DiarizerLoadingTests(unittest.TestCase):
    def test_ui_shows_dependency_error_without_auth_warning(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string(
            "import app\n"
            "app._handle_pipeline_error(\"RuntimeError: Diarization dependency "
            "'matplotlib' is missing. Install requirements.txt.\")\n"
        ).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("matplotlib", app.error[0].value)

    def test_official_metadata_loads_with_restricted_unpickler(self):
        before = torch.serialization.get_safe_globals()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.pt"
            torch.save({
                "version": diarizer.TorchVersion("2.6.0"),
                "specifications": diarizer.Specifications(
                    problem=diarizer.Problem.MONO_LABEL_CLASSIFICATION,
                    resolution=diarizer.Resolution.FRAME,
                    duration=10.0,
                ),
            }, path)

            def load_model(*args, **kwargs):
                metadata = torch.load(path, weights_only=True)
                self.assertEqual(metadata["specifications"].duration, 10.0)
                return MagicMock()

            with (
                patch.object(diarizer, "_pipeline", None),
                patch.object(diarizer, "_select_device", return_value="cpu"),
                patch.object(diarizer.PyannotePipeline, "from_pretrained", side_effect=load_model),
            ):
                self.assertIsNotNone(diarizer.get_pipeline("test-token"))
        self.assertEqual(torch.serialization.get_safe_globals(), before)

    def test_missing_dependency_is_not_reported_as_token_failure(self):
        missing = ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        with (
            patch.object(diarizer, "_pipeline", None),
            patch.object(diarizer, "_select_device", return_value="cpu"),
            patch.object(diarizer.PyannotePipeline, "from_pretrained", side_effect=missing),
        ):
            with self.assertRaises(RuntimeError) as raised:
                diarizer.get_pipeline("test-token")
        self.assertIn("dependency 'matplotlib' is missing", str(raised.exception))
        self.assertIn("requirements.txt", str(raised.exception))
        self.assertNotIn("token permissions", str(raised.exception))
        self.assertIs(raised.exception.__cause__, missing)

    def test_model_access_failure_keeps_context_and_redacts_token(self):
        token = "test-secret-token"
        with (
            patch.object(diarizer, "_pipeline", None),
            patch.object(diarizer, "_select_device", return_value="cpu"),
            patch.object(diarizer.PyannotePipeline, "from_pretrained",
                         side_effect=RuntimeError(f"403 forbidden: {token}")),
        ):
            with self.assertRaises(RuntimeError) as raised:
                diarizer.get_pipeline(token)
        self.assertIn("403 forbidden", str(raised.exception))
        self.assertIn("pyannote/segmentation-3.0", str(raised.exception))
        self.assertNotIn(token, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
