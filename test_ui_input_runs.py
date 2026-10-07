"""Regression checks for selecting YouTube and replacing previous results."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

URL = "https://www.youtube.com/shorts/cj_7iC3pk4A?feature=share"
OLD = [{"start": 0, "end": 9, "speaker": "PREVIOUS", "emotion": "sad", "language": "en"}]
NEW = [{"start": 1, "end": 3, "speaker": "YOUTUBE", "emotion": "happy", "language": "hi"}]


class InputRunTests(unittest.TestCase):
    def run_youtube(self, result=NEW, download_ok=True):
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
            app = AppTest.from_string("import app\napp.main()")
            app.session_state["pipeline_result"] = OLD
            app.session_state["pipeline_duration"] = 9
            app.session_state["pipeline_source"] = "Uploaded file: previous.wav"
            app.run()
            app.radio[0].set_value("🔗 YouTube URL").run()
            app.text_input[0].set_value(URL)
            with (
                patch("app.convert_uploaded_file") as upload,
                patch("app.download_youtube_audio", return_value=(
                    Path("cj_7iC3pk4A.wav") if download_ok else None, "cj_7iC3pk4A"
                )) as download,
                patch("app.run_and_show_progress", return_value=result) as pipeline,
            ):
                app.button[-1].click().run()
                self.assertFalse(app.exception)
                upload.assert_not_called()
                download.assert_called_once()
                self.assertEqual(download.call_args.args[0], URL)
                if download_ok:
                    pipeline.assert_called_once()
                else:
                    pipeline.assert_not_called()
            return app

    def test_youtube_results_replace_upload_results_and_show_source(self):
        app = self.run_youtube()
        self.assertEqual(app.session_state["pipeline_result"], NEW)
        self.assertEqual(app.session_state["pipeline_duration"], 3)
        self.assertIn("cj_7iC3pk4A", app.session_state["pipeline_source"])
        self.assertTrue(any("YouTube:" in caption.value for caption in app.caption))

    def test_failed_run_does_not_show_previous_annotations(self):
        for download_ok in (False, True):
            with self.subTest(download_ok=download_ok):
                app = self.run_youtube(result=None, download_ok=download_ok)
                self.assertNotIn("pipeline_result", app.session_state)
                self.assertNotIn("pipeline_source", app.session_state)
                self.assertNotIn("pipeline_duration", app.session_state)
                self.assertFalse(app.dataframe)

    def test_no_speech_shows_current_run_empty_result(self):
        app = self.run_youtube(result=[])
        self.assertEqual(app.session_state["pipeline_result"], [])
        self.assertEqual(app.session_state["pipeline_duration"], 0)
        self.assertTrue(any("No speech" in message.value for message in app.info))
        self.assertFalse(app.dataframe)


if __name__ == "__main__":
    unittest.main()
