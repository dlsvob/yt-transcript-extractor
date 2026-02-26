"""
test_extractor.py — Unit and integration tests for the core extraction module.

Unit tests (fast, no network):
    - URL / ID parsing for every supported format
    - format_text() and format_json() output shape
    - Error cases for malformed input

Integration tests (need network, marked with @pytest.mark.integration):
    - Fetching a transcript from a real YouTube video
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yt_transcript_extractor.errors import (
    VideoNotFoundError,
)
from yt_transcript_extractor.extractor import (
    extract,
    format_doc,
    format_json,
    format_text,
    parse_video_id,
)


# ---------------------------------------------------------------------------
# Helpers — a fake FetchedTranscript for unit-testing formatters
# ---------------------------------------------------------------------------

class FakeSnippet:
    """Mimics FetchedTranscriptSnippet with .text, .start, .duration."""

    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text = text
        self.start = start
        self.duration = duration


def _make_fake_transcript(snippets_data: list[dict]) -> MagicMock:
    """
    Build a mock FetchedTranscript from a list of dicts.

    The mock supports iteration (for format_text) and .to_raw_data()
    (for format_json), matching the real FetchedTranscript interface.
    """
    snippets = [FakeSnippet(**s) for s in snippets_data]
    mock = MagicMock()
    mock.__iter__ = MagicMock(return_value=iter(snippets))
    mock.to_raw_data = MagicMock(return_value=snippets_data)
    return mock


# ---------------------------------------------------------------------------
# parse_video_id — URL parsing
# ---------------------------------------------------------------------------

class TestParseVideoId:
    """Tests for parse_video_id covering every URL format + bare IDs."""

    def test_standard_watch_url(self) -> None:
        """Standard youtube.com/watch?v= URL."""
        assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_url_with_extra_params(self) -> None:
        """Watch URL with additional query parameters like playlist or timestamp."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf&t=42"
        assert parse_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        """youtu.be short-link format."""
        assert parse_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self) -> None:
        """youtube.com/embed/ URL used in iframes."""
        assert parse_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self) -> None:
        """youtube.com/shorts/ URL."""
        assert parse_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id(self) -> None:
        """Raw 11-character video ID with no URL wrapper."""
        assert parse_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id_with_whitespace(self) -> None:
        """Bare ID with leading/trailing spaces should be trimmed."""
        assert parse_video_id("  dQw4w9WgXcQ  ") == "dQw4w9WgXcQ"

    def test_http_without_www(self) -> None:
        """URL with http:// and no www prefix."""
        assert parse_video_id("http://youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self) -> None:
        """Completely unrelated string should raise VideoNotFoundError."""
        with pytest.raises(VideoNotFoundError):
            parse_video_id("not-a-youtube-url")

    def test_empty_string_raises(self) -> None:
        """Empty input should raise VideoNotFoundError."""
        with pytest.raises(VideoNotFoundError):
            parse_video_id("")

    def test_id_with_hyphens_and_underscores(self) -> None:
        """IDs can contain hyphens and underscores (base64url alphabet)."""
        assert parse_video_id("Ab_Cd-Ef_12") == "Ab_Cd-Ef_12"


# ---------------------------------------------------------------------------
# format_text — plain-text output
# ---------------------------------------------------------------------------

class TestFormatText:
    """Tests for the plain-text formatter."""

    def test_joins_lines(self) -> None:
        """Each snippet becomes one line, joined by newlines."""
        transcript = _make_fake_transcript([
            {"text": "Hello world", "start": 0.0, "duration": 1.5},
            {"text": "Second line", "start": 1.5, "duration": 2.0},
        ])
        result = format_text(transcript)
        assert result == "Hello world\nSecond line"

    def test_empty_transcript(self) -> None:
        """An empty transcript produces an empty string."""
        transcript = _make_fake_transcript([])
        assert format_text(transcript) == ""

    def test_single_segment(self) -> None:
        """A transcript with one segment has no trailing newline."""
        transcript = _make_fake_transcript([
            {"text": "Only segment", "start": 0.0, "duration": 3.0},
        ])
        assert format_text(transcript) == "Only segment"


# ---------------------------------------------------------------------------
# format_json — structured JSON output
# ---------------------------------------------------------------------------

class TestFormatJson:
    """Tests for the JSON formatter."""

    def test_structure(self) -> None:
        """Output dict has video_id, segment_count, and segments list."""
        data = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
            {"text": "World", "start": 1.0, "duration": 1.5},
        ]
        transcript = _make_fake_transcript(data)
        result = format_json(transcript, "test_video_id")

        assert result["video_id"] == "test_video_id"
        assert result["segment_count"] == 2
        assert result["segments"] == data

    def test_empty_segments(self) -> None:
        """Empty transcript produces segment_count 0 and empty list."""
        transcript = _make_fake_transcript([])
        result = format_json(transcript, "empty_vid")
        assert result["segment_count"] == 0
        assert result["segments"] == []


# ---------------------------------------------------------------------------
# format_doc — readable markdown document output
# ---------------------------------------------------------------------------

class TestFormatDoc:
    """Tests for the HTML document formatter with collapsible timestamped sections."""

    def test_returns_full_html_document(self) -> None:
        """format_doc() returns a complete HTML document with doctype and head."""
        transcript = _make_fake_transcript([
            {"text": "Hello world", "start": 0.0, "duration": 5.0},
        ])
        result = format_doc(transcript)

        assert result.startswith("<!DOCTYPE html>")
        assert "<html" in result
        assert "</html>" in result
        assert "<head>" in result
        assert "<body>" in result

    def test_title_in_html(self) -> None:
        """The title parameter appears in both <title> and <h1> tags."""
        transcript = _make_fake_transcript([
            {"text": "Hello", "start": 0.0, "duration": 5.0},
        ])
        result = format_doc(transcript, title="My Video")

        assert "<title>My Video</title>" in result
        assert "<h1>My Video</h1>" in result

    def test_groups_segments_into_sections(self) -> None:
        """Segments within the same ~30s window are joined into one collapsible section."""
        transcript = _make_fake_transcript([
            {"text": "Hello world", "start": 0.0, "duration": 5.0},
            {"text": "how are you", "start": 5.0, "duration": 5.0},
            {"text": "doing today", "start": 10.0, "duration": 5.0},
        ])
        result = format_doc(transcript)

        # All three segments are within 30 seconds, so they form one section.
        assert "\"timestamp\">00:00</span>" in result
        assert "Hello world how are you doing today" in result
        assert result.count("<details>") == 1

    def test_new_section_at_30_second_boundary(self) -> None:
        """A new section starts when a segment crosses the 30-second threshold."""
        transcript = _make_fake_transcript([
            {"text": "First part", "start": 0.0, "duration": 10.0},
            {"text": "still first", "start": 10.0, "duration": 10.0},
            {"text": "second part", "start": 31.0, "duration": 10.0},
            {"text": "still second", "start": 35.0, "duration": 10.0},
        ])
        result = format_doc(transcript)

        assert "\"timestamp\">00:00</span>" in result
        assert "First part still first" in result
        assert "\"timestamp\">00:31</span>" in result
        assert "second part still second" in result
        assert result.count("<details>") == 2

    def test_timestamps_format_correctly(self) -> None:
        """Timestamps beyond 60 seconds use correct MM:SS formatting."""
        transcript = _make_fake_transcript([
            {"text": "intro", "start": 0.0, "duration": 5.0},
            {"text": "later", "start": 92.0, "duration": 5.0},
        ])
        result = format_doc(transcript)

        assert "\"timestamp\">00:00</span>" in result
        # 92 seconds = 1 minute 32 seconds
        assert "\"timestamp\">01:32</span>" in result

    def test_single_segment(self) -> None:
        """A transcript with one segment produces one collapsible section."""
        transcript = _make_fake_transcript([
            {"text": "Only segment", "start": 0.0, "duration": 3.0},
        ])
        result = format_doc(transcript)
        assert "\"timestamp\">00:00</span>" in result
        assert "Only segment" in result
        assert result.count("<details>") == 1

    def test_empty_transcript(self) -> None:
        """An empty transcript produces an empty string."""
        transcript = _make_fake_transcript([])
        assert format_doc(transcript) == ""

    def test_accepts_list_of_dicts(self) -> None:
        """format_doc() also works with plain dicts (from stored segments)."""
        segments = [
            {"text": "Hello", "start": 0.0, "duration": 5.0},
            {"text": "World", "start": 5.0, "duration": 5.0},
        ]
        # Pass the list directly — not wrapped in a mock.
        result = format_doc(segments)
        assert "\"timestamp\">00:00</span>" in result
        assert "Hello World" in result

    def test_multiple_section_boundaries(self) -> None:
        """Multiple 30-second boundaries produce multiple collapsible sections."""
        transcript = _make_fake_transcript([
            {"text": "first", "start": 0.0, "duration": 5.0},
            {"text": "second", "start": 35.0, "duration": 5.0},
            {"text": "third", "start": 70.0, "duration": 5.0},
        ])
        result = format_doc(transcript)

        assert "\"timestamp\">00:00</span>" in result
        assert "\"timestamp\">00:35</span>" in result
        assert "\"timestamp\">01:10</span>" in result
        assert result.count("<details>") == 3
        assert result.count("</details>") == 3


# ---------------------------------------------------------------------------
# Integration tests — require network access
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that hit YouTube's servers.

    Run with:  uv run pytest -m integration
    Skipped by default in CI; use the marker to opt in.
    """

    # "Never Gonna Give You Up" — one of the most stable videos on YouTube,
    # virtually guaranteed to have English captions.
    VIDEO_ID = "dQw4w9WgXcQ"

    def test_extract_text(self) -> None:
        """Fetching a real video in text format returns non-empty text."""
        result = extract(self.VIDEO_ID, fmt="text")
        assert isinstance(result, str)
        assert len(result) > 100  # a real transcript is at least a few hundred chars

    def test_extract_json(self) -> None:
        """Fetching a real video in JSON format returns structured data."""
        result = extract(self.VIDEO_ID, fmt="json")
        assert isinstance(result, dict)
        assert result["video_id"] == self.VIDEO_ID
        assert result["segment_count"] > 0
        assert len(result["segments"]) == result["segment_count"]


# ---------------------------------------------------------------------------
# extract() with save=True — mocked storage and metadata
# ---------------------------------------------------------------------------

class TestExtractWithSave:
    """
    Tests for the save=True path in extract().

    Both the transcript fetch and metadata fetch are mocked so these tests
    run without network access.  We verify that save_transcript() is called
    with the correct arguments.
    """

    @patch("yt_transcript_extractor.extractor.get_transcript")
    @patch("yt_transcript_extractor.metadata.fetch_video_metadata")
    @patch("yt_transcript_extractor.storage.TranscriptStore")
    def test_save_calls_store(
        self,
        MockStore: MagicMock,
        mock_metadata: MagicMock,
        mock_get_transcript: MagicMock,
    ) -> None:
        """extract(save=True) fetches metadata and calls save_transcript()."""
        # Set up mock transcript.
        fake_transcript = _make_fake_transcript([
            {"text": "Hello", "start": 0.0, "duration": 1.0},
        ])
        mock_get_transcript.return_value = fake_transcript

        # Set up mock metadata.
        from yt_transcript_extractor.metadata import VideoMetadata
        from datetime import date
        mock_metadata.return_value = VideoMetadata(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            channel_id="UC_test",
            channel_name="Test Channel",
            channel_url=None,
            upload_date=date(2020, 1, 1),
            duration_secs=60,
        )

        # Set up mock store context manager.
        mock_store_instance = MagicMock()
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store_instance)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        result = extract("dQw4w9WgXcQ", save=True, db_path="/tmp/test.duckdb")

        # Verify metadata was fetched.
        mock_metadata.assert_called_once_with("dQw4w9WgXcQ")
        # Verify store was opened with the correct path.
        MockStore.assert_called_once_with("/tmp/test.duckdb")
        # Verify save_transcript was called.
        mock_store_instance.save_transcript.assert_called_once()

    @patch("yt_transcript_extractor.extractor.get_transcript")
    def test_save_false_skips_storage(self, mock_get_transcript: MagicMock) -> None:
        """extract(save=False) does not import or call storage modules."""
        fake_transcript = _make_fake_transcript([
            {"text": "Hello", "start": 0.0, "duration": 1.0},
        ])
        mock_get_transcript.return_value = fake_transcript

        # This should work without any storage/metadata mocks because
        # save=False means those code paths are never reached.
        result = extract("dQw4w9WgXcQ", save=False)
        assert isinstance(result, str)
