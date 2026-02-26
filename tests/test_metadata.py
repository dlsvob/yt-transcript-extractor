"""
test_metadata.py — Tests for the video metadata fetching module.

All tests mock yt_dlp.YoutubeDL so they run fast without network access.
Covers success paths (full metadata, missing optional fields) and error
paths (DownloadError, None info_dict).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from yt_transcript_extractor.errors import MetadataFetchError
from yt_transcript_extractor.metadata import VideoMetadata, fetch_video_metadata


# ---------------------------------------------------------------------------
# Helpers — sample yt-dlp info dicts
# ---------------------------------------------------------------------------

def _make_info_dict(**overrides) -> dict:
    """
    Build a realistic yt-dlp info_dict with sensible defaults.

    Any key can be overridden via keyword arguments.  This avoids repeating
    the full dict in every test.
    """
    base = {
        "id": "dQw4w9WgXcQ",
        "title": "Rick Astley - Never Gonna Give You Up",
        "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
        "channel": "Rick Astley",
        "uploader": "Rick Astley",
        "channel_url": "https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw",
        "upload_date": "20091025",
        "duration": 213,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

class TestFetchVideoMetadata:
    """Tests for fetch_video_metadata() with mocked yt-dlp."""

    @patch("yt_transcript_extractor.metadata.yt_dlp")
    def test_full_metadata(self, mock_yt_dlp: MagicMock) -> None:
        """All fields are populated when yt-dlp returns a complete info_dict."""
        # Set up the mock: YoutubeDL() returns a context manager whose
        # extract_info() returns our fake info dict.
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = _make_info_dict()
        mock_yt_dlp.YoutubeDL.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.YoutubeDL.return_value.__exit__ = MagicMock(return_value=False)

        result = fetch_video_metadata("dQw4w9WgXcQ")

        assert isinstance(result, VideoMetadata)
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.title == "Rick Astley - Never Gonna Give You Up"
        assert result.channel_id == "UCuAXFkgsw1L7xaCfnd5JJOw"
        assert result.channel_name == "Rick Astley"
        assert result.channel_url == "https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw"
        assert result.upload_date == date(2009, 10, 25)
        assert result.duration_secs == 213

    @patch("yt_transcript_extractor.metadata.yt_dlp")
    def test_missing_upload_date(self, mock_yt_dlp: MagicMock) -> None:
        """upload_date is None when yt-dlp doesn't provide one (e.g. livestreams)."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = _make_info_dict(upload_date=None)
        mock_yt_dlp.YoutubeDL.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.YoutubeDL.return_value.__exit__ = MagicMock(return_value=False)

        result = fetch_video_metadata("dQw4w9WgXcQ")

        assert result.upload_date is None

    @patch("yt_transcript_extractor.metadata.yt_dlp")
    def test_missing_duration(self, mock_yt_dlp: MagicMock) -> None:
        """duration_secs is None when yt-dlp doesn't provide a duration."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = _make_info_dict(duration=None)
        mock_yt_dlp.YoutubeDL.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.YoutubeDL.return_value.__exit__ = MagicMock(return_value=False)

        result = fetch_video_metadata("dQw4w9WgXcQ")

        assert result.duration_secs is None

    @patch("yt_transcript_extractor.metadata.yt_dlp")
    def test_falls_back_to_uploader(self, mock_yt_dlp: MagicMock) -> None:
        """channel_name falls back to 'uploader' when 'channel' key is missing."""
        info = _make_info_dict()
        del info["channel"]  # remove the primary channel name key
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = info
        mock_yt_dlp.YoutubeDL.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.YoutubeDL.return_value.__exit__ = MagicMock(return_value=False)

        result = fetch_video_metadata("dQw4w9WgXcQ")

        # Should fall back to the "uploader" field.
        assert result.channel_name == "Rick Astley"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestFetchVideoMetadataErrors:
    """Tests for error handling in fetch_video_metadata()."""

    @patch("yt_transcript_extractor.metadata.yt_dlp")
    def test_download_error_raises_metadata_fetch_error(self, mock_yt_dlp: MagicMock) -> None:
        """yt-dlp DownloadError is wrapped in MetadataFetchError."""
        # Create a real-looking DownloadError.
        mock_yt_dlp.utils.DownloadError = type("DownloadError", (Exception,), {})
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = mock_yt_dlp.utils.DownloadError("Video not found")
        mock_yt_dlp.YoutubeDL.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.YoutubeDL.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(MetadataFetchError) as exc_info:
            fetch_video_metadata("badid123456")

        assert "badid123456" in exc_info.value.message
        assert exc_info.value.http_status == 502

    @patch("yt_transcript_extractor.metadata.yt_dlp")
    def test_none_info_raises_metadata_fetch_error(self, mock_yt_dlp: MagicMock) -> None:
        """extract_info returning None raises MetadataFetchError."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = None
        mock_yt_dlp.YoutubeDL.return_value.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.YoutubeDL.return_value.__exit__ = MagicMock(return_value=False)
        # Ensure DownloadError is defined so the except clause works.
        mock_yt_dlp.utils.DownloadError = type("DownloadError", (Exception,), {})

        with pytest.raises(MetadataFetchError) as exc_info:
            fetch_video_metadata("dQw4w9WgXcQ")

        assert "no info" in exc_info.value.message.lower()
