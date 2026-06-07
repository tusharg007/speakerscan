"""
Streamlit-free unit tests for UI helper functions.

Tests formatting, demo loading, chart generation, and summary computation
without requiring a Streamlit runtime context.

Usage::

    python test_streamlit.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))


def test_format_time() -> None:
    """Test format_time with various inputs."""
    from ui_helpers import format_time

    cases = [
        (0, "0:00"),
        (5, "0:05"),
        (59, "0:59"),
        (60, "1:00"),
        (65, "1:05"),
        (600, "10:00"),
        (3600, "1:00:00"),
        (3661, "1:01:01"),
        (7384, "2:03:04"),
        (-5, "0:00"),
        (0.7, "0:00"),
        (59.9, "0:59"),
    ]

    passed = 0
    for seconds, expected in cases:
        result = format_time(seconds)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            print(f"  {status}: format_time({seconds}) = '{result}', expected '{expected}'")
        else:
            passed += 1

    print(f"format_time: {passed}/{len(cases)} passed")
    assert passed == len(cases), f"format_time: {len(cases) - passed} failures"


def test_format_confidence() -> None:
    """Test format_confidence percentage formatting."""
    from ui_helpers import format_confidence

    cases = [
        (0.0, "0.0%"),
        (0.5, "50.0%"),
        (0.847, "84.7%"),
        (1.0, "100.0%"),
        (0.999, "99.9%"),
    ]

    passed = 0
    for value, expected in cases:
        result = format_confidence(value)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: format_confidence({value}) = '{result}', expected '{expected}'")

    print(f"format_confidence: {passed}/{len(cases)} passed")
    assert passed == len(cases)


def test_load_demo_sample_exists() -> None:
    """Test loading a demo JSON file that exists."""
    from ui_helpers import load_demo_sample

    # Create a temp directory with a demo file
    with tempfile.TemporaryDirectory() as tmp:
        demo_dir = Path(tmp)
        sample_data = [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 3.5,
                "emotion": "neutral",
                "emotion_confidence": 0.9,
                "language": "en",
                "language_confidence": 0.95,
            }
        ]
        demo_file = demo_dir / "demo_1.json"
        demo_file.write_text(json.dumps(sample_data), encoding="utf-8")

        result = load_demo_sample(demo_dir, "demo_1.json")
        assert result is not None, "Should load existing demo file"
        assert len(result) == 1, f"Expected 1 segment, got {len(result)}"
        assert result[0]["speaker"] == "SPEAKER_00"

    print("load_demo_sample (exists): PASS")


def test_load_demo_sample_missing() -> None:
    """Test loading a demo JSON file that doesn't exist."""
    from ui_helpers import load_demo_sample

    with tempfile.TemporaryDirectory() as tmp:
        result = load_demo_sample(Path(tmp), "nonexistent.json")
        assert result is None, "Should return None for missing file"

    print("load_demo_sample (missing): PASS")


def test_compute_summary_metrics() -> None:
    """Test summary metric computation."""
    from ui_helpers import compute_summary_metrics

    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "emotion": "neutral", "language": "en"},
        {"speaker": "SPEAKER_00", "start": 6.0, "end": 10.0, "emotion": "happy", "language": "hi"},
        {"speaker": "SPEAKER_01", "start": 11.0, "end": 15.0, "emotion": "neutral", "language": "hi"},
        {"speaker": "SPEAKER_01", "start": 16.0, "end": 20.0, "emotion": "neutral", "language": "hi"},
    ]

    metrics = compute_summary_metrics(segments, 20.0)
    assert metrics["num_speakers"] == 2, f"Expected 2 speakers, got {metrics['num_speakers']}"
    assert metrics["dominant_emotion"] == "neutral", f"Expected neutral, got {metrics['dominant_emotion']}"
    assert metrics["dominant_language"] == "hi", f"Expected hi, got {metrics['dominant_language']}"
    assert metrics["duration"] == "0:20"

    # Empty segments
    empty_metrics = compute_summary_metrics([], 0)
    assert empty_metrics["num_speakers"] == 0
    assert empty_metrics["dominant_language"] == "N/A"

    print("compute_summary_metrics: PASS")


def test_build_speaker_timeline() -> None:
    """Test Plotly timeline chart generation with mock data."""
    from ui_helpers import build_speaker_timeline

    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "emotion": "neutral",
         "emotion_confidence": 0.9, "language": "en", "language_confidence": 0.95},
        {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.0, "emotion": "happy",
         "emotion_confidence": 0.7, "language": "en", "language_confidence": 0.88},
        {"speaker": "SPEAKER_00", "start": 7.5, "end": 12.0, "emotion": "angry",
         "emotion_confidence": 0.6, "language": "hi", "language_confidence": 0.75},
    ]

    fig = build_speaker_timeline(segments)
    assert fig is not None, "Should return a figure"

    # Check it has data
    assert len(fig.data) > 0, "Figure should have data traces"

    # Empty segments
    fig_empty = build_speaker_timeline([])
    assert fig_empty is None, "Should return None for empty segments"

    print("build_speaker_timeline: PASS")


def test_segments_to_dataframe() -> None:
    """Test segment-to-DataFrame conversion."""
    from ui_helpers import segments_to_dataframe

    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.5, "emotion": "neutral",
         "emotion_confidence": 0.9, "language": "en", "language_confidence": 0.95},
    ]

    df = segments_to_dataframe(segments)
    assert len(df) == 1
    assert "Speaker" in df.columns
    assert "Emotion" in df.columns
    assert df.iloc[0]["Speaker"] == "SPEAKER_00"
    assert df.iloc[0]["Start"] == "0:00"
    assert df.iloc[0]["End"] == "0:05"

    print("segments_to_dataframe: PASS")


def test_segments_to_csv_bytes() -> None:
    """Test CSV export."""
    from ui_helpers import segments_to_csv_bytes

    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "emotion": "neutral",
         "emotion_confidence": 0.85, "language": "en", "language_confidence": 0.9},
    ]

    csv_data = segments_to_csv_bytes(segments)
    assert isinstance(csv_data, bytes)
    text = csv_data.decode("utf-8")
    assert "Speaker" in text
    assert "SPEAKER_00" in text

    print("segments_to_csv_bytes: PASS")


def main() -> None:
    """Run all tests."""
    print("=" * 50)
    print("Speech Pipeline — Streamlit Helper Tests")
    print("=" * 50)
    print()

    tests = [
        test_format_time,
        test_format_confidence,
        test_load_demo_sample_exists,
        test_load_demo_sample_missing,
        test_compute_summary_metrics,
        test_build_speaker_timeline,
        test_segments_to_dataframe,
        test_segments_to_csv_bytes,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL: {test_fn.__name__}: {exc}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
