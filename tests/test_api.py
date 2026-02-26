"""
test_api.py — Tests for the FastAPI web API endpoints.

Uses FastAPI's TestClient (backed by httpx) so tests run in-process without
needing a live server.  Transcript fetching and storage are mocked so these
tests are fast and don't require network access or a real database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from yt_transcript_extractor.api import app
from yt_transcript_extractor.errors import (
    LanguageNotAvailableError,
    TranscriptUnavailableError,
    VideoNotFoundError,
)
from yt_transcript_extractor.storage import ChannelRecord, VideoRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    """Create a fresh TestClient for each test."""
    return TestClient(app)


# Sample data returned by mocked extract() calls.
_SAMPLE_TEXT = "Hello world\nSecond line"
_SAMPLE_JSON = {
    "video_id": "dQw4w9WgXcQ",
    "segment_count": 2,
    "segments": [
        {"text": "Hello world", "start": 0.0, "duration": 1.5},
        {"text": "Second line", "start": 1.5, "duration": 2.0},
    ],
}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health endpoint returns 200 with status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Transcript endpoint — success cases
# ---------------------------------------------------------------------------

class TestTranscriptEndpoint:
    """Tests for GET /transcript/{video_id} with mocked extraction."""

    @patch("yt_transcript_extractor.api.extract")
    def test_text_format(self, mock_extract: MagicMock, client: TestClient) -> None:
        """Default format=text returns plain text with 200."""
        mock_extract.return_value = _SAMPLE_TEXT

        resp = client.get("/transcript/dQw4w9WgXcQ")

        assert resp.status_code == 200
        assert resp.text == _SAMPLE_TEXT
        # Verify extract() was called with the right arguments.
        # save defaults to False, db_path is None when save is False.
        mock_extract.assert_called_once_with(
            "dQw4w9WgXcQ", languages=None, fmt="text",
            save=False, db_path=None,
        )

    @patch("yt_transcript_extractor.api.extract")
    def test_json_format(self, mock_extract: MagicMock, client: TestClient) -> None:
        """format=json returns JSON body with 200."""
        mock_extract.return_value = _SAMPLE_JSON

        resp = client.get("/transcript/dQw4w9WgXcQ?format=json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["video_id"] == "dQw4w9WgXcQ"
        assert data["segment_count"] == 2

    @patch("yt_transcript_extractor.api.extract")
    def test_language_param(self, mock_extract: MagicMock, client: TestClient) -> None:
        """The lang query param is split and forwarded to extract()."""
        mock_extract.return_value = _SAMPLE_TEXT

        resp = client.get("/transcript/dQw4w9WgXcQ?lang=de,en")

        assert resp.status_code == 200
        mock_extract.assert_called_once_with(
            "dQw4w9WgXcQ", languages=["de", "en"], fmt="text",
            save=False, db_path=None,
        )

    @patch("yt_transcript_extractor.api.extract")
    def test_save_param_passed_to_extract(self, mock_extract: MagicMock, client: TestClient) -> None:
        """The save=true query param enables transcript persistence."""
        mock_extract.return_value = _SAMPLE_TEXT

        resp = client.get("/transcript/dQw4w9WgXcQ?save=true")

        assert resp.status_code == 200
        mock_extract.assert_called_once_with(
            "dQw4w9WgXcQ", languages=None, fmt="text",
            save=True, db_path="transcripts.duckdb",
        )

    def test_invalid_format_returns_422(self, client: TestClient) -> None:
        """An unsupported format value is rejected by FastAPI validation (422)."""
        resp = client.get("/transcript/dQw4w9WgXcQ?format=xml")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Transcript endpoint — error cases
# ---------------------------------------------------------------------------

class TestTranscriptErrors:
    """Tests that TranscriptError subclasses produce the correct HTTP status."""

    @patch("yt_transcript_extractor.api.extract")
    def test_video_not_found(self, mock_extract: MagicMock, client: TestClient) -> None:
        """VideoNotFoundError → HTTP 404."""
        mock_extract.side_effect = VideoNotFoundError("badid1234ab")

        resp = client.get("/transcript/badid1234ab")

        assert resp.status_code == 404
        assert "error" in resp.json()

    @patch("yt_transcript_extractor.api.extract")
    def test_transcript_unavailable(self, mock_extract: MagicMock, client: TestClient) -> None:
        """TranscriptUnavailableError → HTTP 404."""
        mock_extract.side_effect = TranscriptUnavailableError("dQw4w9WgXcQ")

        resp = client.get("/transcript/dQw4w9WgXcQ")

        assert resp.status_code == 404

    @patch("yt_transcript_extractor.api.extract")
    def test_language_not_available(self, mock_extract: MagicMock, client: TestClient) -> None:
        """LanguageNotAvailableError → HTTP 400."""
        mock_extract.side_effect = LanguageNotAvailableError("dQw4w9WgXcQ", ["xx"])

        resp = client.get("/transcript/dQw4w9WgXcQ?lang=xx")

        assert resp.status_code == 400
        assert "error" in resp.json()


# ---------------------------------------------------------------------------
# Channels endpoint
# ---------------------------------------------------------------------------

class TestChannelsEndpoint:
    """Tests for GET /channels with mocked TranscriptStore."""

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_list_channels(self, MockStore: MagicMock, client: TestClient) -> None:
        """Returns channel list with video counts."""
        mock_store = MagicMock()
        mock_store.list_channels.return_value = [
            ChannelRecord(
                channel_id="UC_test",
                channel_name="Test Channel",
                channel_url="https://youtube.com/channel/UC_test",
                video_count=3,
            ),
        ]
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/channels")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["channels"]) == 1
        assert data["channels"][0]["channel_name"] == "Test Channel"
        assert data["channels"][0]["video_count"] == 3

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_list_channels_empty(self, MockStore: MagicMock, client: TestClient) -> None:
        """Returns empty list when no channels are saved."""
        mock_store = MagicMock()
        mock_store.list_channels.return_value = []
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/channels")

        assert resp.status_code == 200
        assert resp.json()["channels"] == []


# ---------------------------------------------------------------------------
# Videos endpoint
# ---------------------------------------------------------------------------

class TestVideosEndpoint:
    """Tests for GET /channels/{channel_id}/videos."""

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_list_videos(self, MockStore: MagicMock, client: TestClient) -> None:
        """Returns video list for a channel."""
        from datetime import date, datetime
        mock_store = MagicMock()
        mock_store.list_videos.return_value = [
            VideoRecord(
                video_id="dQw4w9WgXcQ",
                title="Never Gonna Give You Up",
                channel_id="UC_test",
                upload_date=date(2009, 10, 25),
                duration_secs=213,
                language="English",
                language_code="en",
                is_generated=False,
                created_at=datetime(2024, 1, 1),
            ),
        ]
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/channels/UC_test/videos")

        assert resp.status_code == 200
        data = resp.json()
        assert data["channel_id"] == "UC_test"
        assert len(data["videos"]) == 1
        assert data["videos"][0]["video_id"] == "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Saved transcript endpoint
# ---------------------------------------------------------------------------

class TestSavedEndpoint:
    """Tests for GET /saved/{video_id}."""

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_saved_text(self, MockStore: MagicMock, client: TestClient) -> None:
        """Returns plain text for a saved transcript."""
        mock_store = MagicMock()
        mock_store.has_video.return_value = True
        mock_store.get_transcript_text.return_value = "Hello world\nSecond line"
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/saved/dQw4w9WgXcQ")

        assert resp.status_code == 200
        assert resp.text == "Hello world\nSecond line"

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_saved_json(self, MockStore: MagicMock, client: TestClient) -> None:
        """Returns JSON for a saved transcript when format=json."""
        mock_store = MagicMock()
        mock_store.has_video.return_value = True
        mock_store.get_transcript.return_value = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
        ]
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/saved/dQw4w9WgXcQ?format=json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["video_id"] == "dQw4w9WgXcQ"
        assert data["segment_count"] == 1

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_saved_not_found(self, MockStore: MagicMock, client: TestClient) -> None:
        """Returns 404 when the video isn't in the database."""
        mock_store = MagicMock()
        mock_store.has_video.return_value = False
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/saved/nonexistent1")

        assert resp.status_code == 404
        assert "error" in resp.json()


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    """Tests for GET /search."""

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_search_returns_results(self, MockStore: MagicMock, client: TestClient) -> None:
        """Search returns matching segments with video context."""
        mock_store = MagicMock()
        mock_store.search_transcripts.return_value = [
            {
                "video_id": "dQw4w9WgXcQ",
                "title": "Never Gonna Give You Up",
                "channel_name": "Rick Astley",
                "seq": 0,
                "text": "Never gonna give you up",
                "start": 0.0,
                "duration": 2.5,
            },
        ]
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/search?q=never+gonna")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "never gonna"
        assert data["result_count"] == 1
        assert data["results"][0]["text"] == "Never gonna give you up"

    @patch("yt_transcript_extractor.api.TranscriptStore")
    def test_search_empty_results(self, MockStore: MagicMock, client: TestClient) -> None:
        """Search returns empty list when nothing matches."""
        mock_store = MagicMock()
        mock_store.search_transcripts.return_value = []
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/search?q=xyznonexistent")

        assert resp.status_code == 200
        data = resp.json()
        assert data["result_count"] == 0
        assert data["results"] == []

    def test_search_missing_query_returns_422(self, client: TestClient) -> None:
        """Missing q parameter returns 422 validation error."""
        resp = client.get("/search")
        assert resp.status_code == 422
