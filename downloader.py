"""
Audio downloader — YouTube + local file support.

Downloads audio from YouTube via ``yt-dlp``, converts to 16 kHz mono WAV
via ``ffmpeg-python``, and validates the result with ``soundfile``.

Features:
    * Idempotent — skips already-downloaded WAVs.
    * Retry with exponential back-off for transient YouTube errors.
    * Local file support — non-URL inputs skip the download stage.
    * Validates output WAV (sample rate, channels, duration).
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import ffmpeg
import soundfile as sf
import yt_dlp
from loguru import logger

from config import (
    MAX_RETRIES,
    MIN_AUDIO_DURATION,
    MONO_CHANNELS,
    RETRY_BACKOFF_BASE,
    SAMPLE_RATE,
)

# ── ffmpeg binary resolution ────────────────────────────────────


def _resolve_ffmpeg() -> str:
    """Resolve the ffmpeg binary path.

    Checks (in order):
        1. System PATH (``shutil.which``).
        2. ``imageio-ffmpeg`` bundled binary.

    Returns:
        Absolute path to the ffmpeg executable.

    Raises:
        FileNotFoundError: If ffmpeg cannot be found anywhere.
    """
    # 1. System PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # 2. imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            logger.debug("Using imageio-ffmpeg binary: {}", bundled)
            return bundled
    except ImportError:
        pass

    raise FileNotFoundError(
        "ffmpeg not found. Install it via: pip install imageio-ffmpeg  "
        "or install ffmpeg system-wide and add to PATH."
    )


# Cache the resolved path at module load time
_FFMPEG_BIN: str = _resolve_ffmpeg()


# ── public API ──────────────────────────────────────────────────


def download_audio(
    url_or_path: str,
    raw_dir: Path,
    audio_dir: Path,
    file_id: str | None = None,
) -> Path | None:
    """Download or locate source audio and convert to 16 kHz mono WAV.

    Args:
        url_or_path: A YouTube URL **or** a local file path.
        raw_dir: Directory for intermediate downloaded files.
        audio_dir: Directory for the final WAV output.
        file_id: Override for the file identifier. If ``None``, derived
                 automatically from the URL or filename.

    Returns:
        The ``Path`` to the output WAV file, or ``None`` on failure.
    """
    file_id = file_id or extract_file_id(url_or_path)
    wav_path = audio_dir / f"{file_id}.wav"

    # Idempotency — skip if WAV already exists and is valid
    if wav_path.exists() and _validate_wav(wav_path):
        logger.info("[{}] WAV already exists — skipping download", file_id)
        return wav_path

    start = time.perf_counter()
    is_url = _is_url(url_or_path)

    try:
        if is_url:
            raw_path = _download_with_retry(url_or_path, raw_dir, file_id)
        else:
            raw_path = Path(url_or_path)
            if not raw_path.exists():
                logger.error("[{}] Local file not found: {}", file_id, raw_path)
                return None

        wav_path = _convert_to_wav(raw_path, wav_path)

        if not _validate_wav(wav_path):
            logger.error("[{}] Output WAV validation failed", file_id)
            return None

        # Clean up raw download (keep local source files untouched)
        if is_url and raw_path.exists():
            raw_path.unlink(missing_ok=True)
            logger.debug("[{}] Cleaned up raw file {}", file_id, raw_path.name)

        elapsed = time.perf_counter() - start
        logger.info("[{}] Download + convert completed in {:.1f}s", file_id, elapsed)
        return wav_path

    except Exception as exc:
        logger.error("[{}] Download pipeline failed: {}", file_id, exc)
        return None


# ── file-ID extraction ──────────────────────────────────────────


def extract_file_id(url_or_path: str) -> str:
    """Derive a short, filesystem-safe identifier from a URL or file path.

    For YouTube URLs, extracts the 11-character video ID.
    For local paths, uses the stem of the filename.
    """
    if _is_url(url_or_path):
        return _extract_youtube_id(url_or_path)
    return Path(url_or_path).stem


def _extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats.

    Handles:
        * ``https://www.youtube.com/watch?v=ID``
        * ``https://youtu.be/ID``
        * ``https://www.youtube.com/embed/ID``
        * ``https://www.youtube.com/shorts/ID``
    """
    parsed = urlparse(url)

    # Standard watch URL
    if "youtube.com" in parsed.hostname or "youtube-nocookie.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0][:11]
        # embed / shorts / live
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            return parts[-1][:11]

    # Short URL
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return parsed.path.strip("/")[:11]

    # Fallback — hash the URL for a stable ID
    logger.warning("Could not parse YouTube ID from {}; using hash fallback", url)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", url)[:32]


def _is_url(s: str) -> bool:
    """Return ``True`` if *s* looks like an HTTP(S) URL."""
    return s.startswith("http://") or s.startswith("https://")


# ── download with retry ─────────────────────────────────────────


def _download_with_retry(
    url: str,
    raw_dir: Path,
    file_id: str,
    max_retries: int = MAX_RETRIES,
) -> Path:
    """Download audio from YouTube with exponential back-off.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    ydl_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(raw_dir / f"{file_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,       # yt-dlp internal retries (per fragment)
        "noprogress": True,
    }

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "[{}] Downloading (attempt {}/{})", file_id, attempt, max_retries
            )
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                ext = info.get("ext", "webm")
                raw_path = raw_dir / f"{file_id}.{ext}"

                # yt-dlp may use a different extension than reported
                if not raw_path.exists():
                    candidates = list(raw_dir.glob(f"{file_id}.*"))
                    candidates = [
                        c for c in candidates if c.suffix.lower() != ".wav"
                    ]
                    if candidates:
                        raw_path = candidates[0]
                    else:
                        raise FileNotFoundError(
                            f"yt-dlp reported success but no file found for {file_id}"
                        )

                logger.debug("[{}] Downloaded raw file: {}", file_id, raw_path.name)
                return raw_path

        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "[{}] Attempt {} failed ({}). Retrying in {:.0f}s…",
                    file_id,
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[{}] All {} download attempts exhausted", file_id, max_retries
                )

    raise RuntimeError(
        f"Failed to download {url} after {max_retries} attempts: {last_error}"
    )


# ── ffmpeg conversion ────────────────────────────────────────────


def _convert_to_wav(input_path: Path, wav_path: Path) -> Path:
    """Convert any audio/video file to 16 kHz mono WAV via ffmpeg-python.

    Args:
        input_path: Source audio/video file.
        wav_path: Destination WAV path.

    Returns:
        The *wav_path* on success.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero status.
    """
    logger.debug("Converting {} → {}", input_path.name, wav_path.name)
    try:
        (
            ffmpeg
            .input(str(input_path))
            .output(
                str(wav_path),
                ar=SAMPLE_RATE,
                ac=MONO_CHANNELS,
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(cmd=_FFMPEG_BIN, capture_stdout=True, capture_stderr=True)
        )
        return wav_path
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}") from exc


# ── WAV validation ───────────────────────────────────────────────


def _validate_wav(wav_path: Path) -> bool:
    """Validate that a WAV file meets pipeline requirements.

    Checks:
        * File is readable by soundfile.
        * Sample rate == 16 000 Hz.
        * Channels == 1 (mono).
        * Duration >= ``MIN_AUDIO_DURATION``.

    Returns:
        ``True`` if all checks pass; ``False`` otherwise (with warnings logged).
    """
    try:
        info = sf.info(str(wav_path))
    except Exception as exc:
        logger.warning("Cannot read WAV {}: {}", wav_path.name, exc)
        return False

    ok = True

    if info.samplerate != SAMPLE_RATE:
        logger.warning(
            "{}: expected {}Hz, got {}Hz",
            wav_path.name,
            SAMPLE_RATE,
            info.samplerate,
        )
        ok = False

    if info.channels != MONO_CHANNELS:
        logger.warning(
            "{}: expected mono, got {} channels", wav_path.name, info.channels
        )
        ok = False

    duration = info.frames / info.samplerate
    if duration < MIN_AUDIO_DURATION:
        logger.warning(
            "{}: audio too short ({:.2f}s < {:.1f}s minimum)",
            wav_path.name,
            duration,
            MIN_AUDIO_DURATION,
        )
        ok = False

    if duration < 30.0:
        logger.warning(
            "{}: short audio ({:.1f}s) — diarization may be unreliable",
            wav_path.name,
            duration,
        )
        # Still return True — we process short files with a warning

    return ok
