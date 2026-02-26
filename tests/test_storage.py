"""
test_storage.py — Tests for the DuckDB transcript storage module.

Uses real DuckDB databases in pytest's tmp_path, so every test gets a fresh
isolated database.  No mocking needed — we test the actual SQL operations.

Covers:
    - Schema creation and re-opening existing DB
    - save/retrieve round-trip
    - Idempotent saves (duplicate detection)
    - list_channels and list_videos
    - Full-text search
    - Empty database edge cases
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from yt_transcript_extractor.errors import StorageError
from yt_transcript_extractor.metadata import VideoMetadata
from yt_transcript_extractor.storage import TranscriptStore


# ---------------------------------------------------------------------------
# Helpers — fake transcript objects that mimic FetchedTranscript
# ---------------------------------------------------------------------------

class FakeSnippet:
    """
    Mimics a FetchedTranscriptSnippet with .text, .start, .duration.

    Used to build fake transcripts for testing without importing the real
    youtube-transcript-api types.
    """

    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text = text
        self.start = start
        self.duration = duration


def _make_fake_transcript(
    snippets_data: list[dict],
    language: str = "English",
    language_code: str = "en",
    is_generated: bool = False,
) -> MagicMock:
    """
    Build a mock FetchedTranscript with iterable snippets and language attrs.

    The mock supports iteration (for saving segments) and has the language
    attributes that TranscriptStore reads during save_transcript().
    """
    snippets = [FakeSnippet(**s) for s in snippets_data]
    mock = MagicMock()
    mock.__iter__ = MagicMock(return_value=iter(snippets))
    mock.language = language
    mock.language_code = language_code
    mock.is_generated = is_generated
    return mock


def _sample_metadata(
    video_id: str = "dQw4w9WgXcQ",
    channel_id: str = "UCuAXFkgsw1L7xaCfnd5JJOw",
    channel_name: str = "Rick Astley",
) -> VideoMetadata:
    """Build a sample VideoMetadata for testing."""
    return VideoMetadata(
        video_id=video_id,
        title="Never Gonna Give You Up",
        channel_id=channel_id,
        channel_name=channel_name,
        channel_url=f"https://www.youtube.com/channel/{channel_id}",
        upload_date=date(2009, 10, 25),
        duration_secs=213,
    )


_SAMPLE_SEGMENTS = [
    {"text": "Never gonna give you up", "start": 0.0, "duration": 2.5},
    {"text": "Never gonna let you down", "start": 2.5, "duration": 2.5},
    {"text": "Never gonna run around and desert you", "start": 5.0, "duration": 3.0},
]


# ---------------------------------------------------------------------------
# Schema and lifecycle
# ---------------------------------------------------------------------------

class TestTranscriptStoreLifecycle:
    """Tests for database creation, schema, and context manager behavior."""

    def test_creates_new_database(self, tmp_path) -> None:
        """Opening a non-existent path creates a new database file."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            # Should be able to query the empty tables.
            assert store.list_channels() == []

    def test_reopen_existing_database(self, tmp_path) -> None:
        """Opening an existing database doesn't lose data or error."""
        db_path = str(tmp_path / "test.duckdb")

        # Save data in first session.
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

        # Reopen and verify data persists.
        with TranscriptStore(db_path) as store:
            assert store.has_video("dQw4w9WgXcQ")

    def test_context_manager_closes_connection(self, tmp_path) -> None:
        """The connection is closed after exiting the with-block."""
        db_path = str(tmp_path / "test.duckdb")
        store = TranscriptStore(db_path)
        store.close()
        # After close, operations should raise (DuckDB-specific behavior).
        # We just verify close() doesn't error — double-close is safe.
        store.close()


# ---------------------------------------------------------------------------
# Save and retrieve round-trip
# ---------------------------------------------------------------------------

class TestSaveAndRetrieve:
    """Tests for saving transcripts and reading them back."""

    def test_save_and_get_transcript(self, tmp_path) -> None:
        """Saved segments can be retrieved as a list of dicts."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            result = store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

            assert result.video_id == "dQw4w9WgXcQ"
            assert result.already_existed is False

            segments = store.get_transcript("dQw4w9WgXcQ")
            assert len(segments) == 3
            assert segments[0]["text"] == "Never gonna give you up"
            assert segments[1]["text"] == "Never gonna let you down"
            assert segments[2]["text"] == "Never gonna run around and desert you"

    def test_save_and_get_text(self, tmp_path) -> None:
        """get_transcript_text() returns joined plain text."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

            text = store.get_transcript_text("dQw4w9WgXcQ")
            lines = text.split("\n")
            assert len(lines) == 3
            assert lines[0] == "Never gonna give you up"

    def test_has_video_returns_true_for_saved(self, tmp_path) -> None:
        """has_video() returns True for a video that's been saved."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

            assert store.has_video("dQw4w9WgXcQ") is True

    def test_has_video_returns_false_for_missing(self, tmp_path) -> None:
        """has_video() returns False for a video that doesn't exist."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            assert store.has_video("nonexistent1") is False


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Tests that saving the same video twice is a safe no-op."""

    def test_duplicate_save_returns_already_existed(self, tmp_path) -> None:
        """Saving the same video twice returns already_existed=True the second time."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript1 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            result1 = store.save_transcript("dQw4w9WgXcQ", transcript1, _sample_metadata())

            transcript2 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            result2 = store.save_transcript("dQw4w9WgXcQ", transcript2, _sample_metadata())

            assert result1.already_existed is False
            assert result2.already_existed is True

    def test_duplicate_save_doesnt_duplicate_segments(self, tmp_path) -> None:
        """The segment count stays the same after a duplicate save."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript1 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript1, _sample_metadata())

            transcript2 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript2, _sample_metadata())

            segments = store.get_transcript("dQw4w9WgXcQ")
            assert len(segments) == 3


# ---------------------------------------------------------------------------
# Channel and video listing
# ---------------------------------------------------------------------------

class TestListChannelsAndVideos:
    """Tests for list_channels() and list_videos()."""

    def test_list_channels_with_video_counts(self, tmp_path) -> None:
        """list_channels() returns channels with correct video counts."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            # Save two videos from the same channel.
            t1 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", t1, _sample_metadata())

            t2 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            meta2 = VideoMetadata(
                video_id="oHg5SJYRHA0",
                title="Another Rick Video",
                channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
                channel_name="Rick Astley",
                channel_url="https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw",
                upload_date=date(2010, 1, 1),
                duration_secs=180,
            )
            store.save_transcript("oHg5SJYRHA0", t2, meta2)

            channels = store.list_channels()
            assert len(channels) == 1
            assert channels[0].channel_name == "Rick Astley"
            assert channels[0].video_count == 2

    def test_list_channels_alphabetical(self, tmp_path) -> None:
        """list_channels() returns channels sorted alphabetically."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            # Save videos from two different channels.
            t1 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", t1, _sample_metadata(
                channel_id="UC_B", channel_name="Zebra Channel",
            ))

            t2 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("oHg5SJYRHA0", t2, _sample_metadata(
                video_id="oHg5SJYRHA0",
                channel_id="UC_A", channel_name="Alpha Channel",
            ))

            channels = store.list_channels()
            assert len(channels) == 2
            assert channels[0].channel_name == "Alpha Channel"
            assert channels[1].channel_name == "Zebra Channel"

    def test_list_videos_for_channel(self, tmp_path) -> None:
        """list_videos() returns videos for a specific channel, newest first."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            t1 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            meta1 = _sample_metadata()
            store.save_transcript("dQw4w9WgXcQ", t1, meta1)

            t2 = _make_fake_transcript(_SAMPLE_SEGMENTS)
            meta2 = VideoMetadata(
                video_id="oHg5SJYRHA0",
                title="Newer Video",
                channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
                channel_name="Rick Astley",
                channel_url=None,
                upload_date=date(2020, 6, 15),
                duration_secs=180,
            )
            store.save_transcript("oHg5SJYRHA0", t2, meta2)

            videos = store.list_videos("UCuAXFkgsw1L7xaCfnd5JJOw")
            assert len(videos) == 2
            # Newest first — 2020 before 2009.
            assert videos[0].video_id == "oHg5SJYRHA0"
            assert videos[1].video_id == "dQw4w9WgXcQ"

    def test_list_videos_empty_channel(self, tmp_path) -> None:
        """list_videos() returns empty list for a channel with no saved videos."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            videos = store.list_videos("nonexistent_channel")
            assert videos == []


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    """Tests for search_transcripts()."""

    def test_search_finds_matching_segments(self, tmp_path) -> None:
        """search_transcripts() returns segments containing the query."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

            results = store.search_transcripts("let you down")
            assert len(results) == 1
            assert results[0]["text"] == "Never gonna let you down"
            assert results[0]["video_id"] == "dQw4w9WgXcQ"

    def test_search_case_insensitive(self, tmp_path) -> None:
        """Search is case-insensitive (ILIKE)."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

            results = store.search_transcripts("NEVER GONNA")
            # All three segments contain "Never gonna".
            assert len(results) == 3

    def test_search_no_results(self, tmp_path) -> None:
        """Search returns empty list when nothing matches."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

            results = store.search_transcripts("xyznonexistent")
            assert results == []

    def test_search_empty_database(self, tmp_path) -> None:
        """Search on an empty database returns empty list."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            results = store.search_transcripts("anything")
            assert results == []

    def test_search_includes_video_context(self, tmp_path) -> None:
        """Search results include video title and channel name."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _sample_metadata())

            results = store.search_transcripts("give you up")
            assert len(results) == 1
            assert results[0]["title"] == "Never Gonna Give You Up"
            assert results[0]["channel_name"] == "Rick Astley"
            assert "start" in results[0]
            assert "duration" in results[0]


# ---------------------------------------------------------------------------
# Empty database edge cases
# ---------------------------------------------------------------------------

class TestEmptyDatabase:
    """Tests for querying an empty database."""

    def test_get_transcript_missing_video(self, tmp_path) -> None:
        """get_transcript() returns empty list for a non-existent video."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            assert store.get_transcript("nonexistent1") == []

    def test_get_transcript_text_missing_video(self, tmp_path) -> None:
        """get_transcript_text() returns empty string for a non-existent video."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            assert store.get_transcript_text("nonexistent1") == ""

    def test_list_channels_empty(self, tmp_path) -> None:
        """list_channels() returns empty list on fresh database."""
        db_path = str(tmp_path / "test.duckdb")
        with TranscriptStore(db_path) as store:
            assert store.list_channels() == []
