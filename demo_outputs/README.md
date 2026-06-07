# Demo Outputs

This directory holds pre-computed pipeline output JSON files for the demo
sample buttons in the Streamlit app.

## Expected files

| File | Description |
|------|-------------|
| `demo_1.json` | Hindi conversation (~45s) |
| `demo_2.json` | English monologue (~30s) |
| `demo_3.json` | Code-switched Hindi-English (~60s) |

## JSON schema

Each file is a JSON array of segment objects:

```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.312,
    "end": 3.768,
    "emotion": "neutral",
    "emotion_confidence": 0.847,
    "language": "en",
    "language_confidence": 0.992
  },
  {
    "speaker": "SPEAKER_01",
    "start": 4.012,
    "end": 6.903,
    "emotion": "happy",
    "emotion_confidence": 0.623,
    "language": "hi",
    "language_confidence": 0.988
  }
]
```

### Required fields per segment

| Field | Type | Description |
|-------|------|-------------|
| `speaker` | string | Speaker label (e.g., `SPEAKER_00`) |
| `start` | float | Segment start time in seconds |
| `end` | float | Segment end time in seconds |
| `emotion` | string | One of: `neutral`, `happy`, `angry`, `sad`, `fear`, `surprise`, `disgust`, `too_short`, `error` |
| `emotion_confidence` | float | 0.0–1.0 confidence score |
| `language` | string | ISO 639-1 code (e.g., `en`, `hi`, `ta`) or `too_short`/`error` |
| `language_confidence` | float | 0.0–1.0 confidence score |

## How to generate

1. Run the pipeline locally on your chosen audio samples:

```bash
python main.py --input my_demo_urls.txt --output ./demo_data
```

2. Copy the output JSON files:

```bash
cp demo_data/annotations/VIDEO_ID.json demo_outputs/demo_1.json
```

3. Commit and push to your HF Spaces repo.
