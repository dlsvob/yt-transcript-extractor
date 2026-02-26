"""
test_cli.py — Tests for the CLI module's new default behaviors and helpers.

Covers:
    - _sanitize_filename() with various unsafe characters
    - _auto_output_path() with a real temp DuckDB database
    - Default option values for `get` (save=True, fmt="doc")
    - Default option values for `saved` (fmt="doc")
    - Auto-path file writing for both `get` and `saved` subcommands
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from yt_transcript_extractor.cli import (
    _auto_output_path,
    _sanitize_filename,
    get,
    main,
    saved,
)
from yt_transcript_extractor.metadata import VideoMetadata
from yt_transcript_extractor.storage import TranscriptStore


# ---------------------------------------------------------------------------
# Helpers — reusable fake transcript builder (same pattern as other test files)
# ---------------------------------------------------------------------------

class FakeSnippet:
    """Mimics a FetchedTranscriptSnippet with .text, .start, .duration."""

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
    """Build a mock FetchedTranscript that supports iteration and language attrs."""
    snippets = [FakeSnippet(**s) for s in snippets_data]
    mock = MagicMock()
    mock.__iter__ = MagicMock(return_value=iter(snippets))
    mock.language = language
    mock.language_code = language_code
    mock.is_generated = is_generated
    return mock


_SAMPLE_SEGMENTS = [
    {"text": "Hello world", "start": 0.0, "duration": 5.0},
    {"text": "Second line", "start": 5.0, "duration": 5.0},
]

_SAMPLE_METADATA = VideoMetadata(
    video_id="dQw4w9WgXcQ",
    title="Never Gonna Give You Up",
    channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
    channel_name="Rick Astley",
    channel_url="https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw",
    upload_date=date(2009, 10, 25),
    duration_secs=213,
)


# ---------------------------------------------------------------------------
# _sanitize_filename — filesystem-safe name generation
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    """Tests for the _sanitize_filename() helper that cleans unsafe characters."""

    def test_replaces_colons(self) -> None:
        """Colons (common in video titles like 'Part 1: Intro') become hyphens."""
        assert _sanitize_filename("Part 1: Intro") == "Part 1- Intro"

    def test_replaces_slashes(self) -> None:
        """Forward slashes (e.g. 'AC/DC') become hyphens."""
        assert _sanitize_filename("AC/DC") == "AC-DC"

    def test_replaces_backslashes(self) -> None:
        """Backslashes become hyphens."""
        assert _sanitize_filename("path\\file") == "path-file"

    def test_replaces_question_marks(self) -> None:
        """Question marks (common in video titles) become hyphens."""
        assert _sanitize_filename("What is this?") == "What is this-"

    def test_replaces_asterisks(self) -> None:
        """Asterisks become hyphens."""
        assert _sanitize_filename("5* review") == "5- review"

    def test_replaces_angle_brackets(self) -> None:
        """Angle brackets become hyphens."""
        assert _sanitize_filename("<hello>") == "-hello-"

    def test_replaces_pipes(self) -> None:
        """Pipe characters become hyphens."""
        assert _sanitize_filename("A | B") == "A - B"

    def test_replaces_double_quotes(self) -> None:
        """Double quotes become hyphens."""
        assert _sanitize_filename('Say "hello"') == "Say -hello-"

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert _sanitize_filename("  hello  ") == "hello"

    def test_strips_leading_trailing_dots(self) -> None:
        """Leading and trailing dots are stripped (hidden files on POSIX)."""
        assert _sanitize_filename(".hidden.") == "hidden"

    def test_multiple_unsafe_chars(self) -> None:
        """Multiple different unsafe characters in one string all get replaced."""
        assert _sanitize_filename('A: B/C\\D?E*F<G>H|I"J') == "A- B-C-D-E-F-G-H-I-J"

    def test_safe_string_unchanged(self) -> None:
        """A string with no unsafe characters comes through unchanged."""
        assert _sanitize_filename("Normal Title - Part 2") == "Normal Title - Part 2"

    def test_empty_string(self) -> None:
        """An empty string produces an empty string."""
        assert _sanitize_filename("") == ""


# ---------------------------------------------------------------------------
# _auto_output_path — automatic output path from DB metadata
# ---------------------------------------------------------------------------

class TestAutoOutputPath:
    """Tests for _auto_output_path() which builds file paths from DB metadata."""

    def test_returns_correct_path(self, tmp_path) -> None:
        """Returns ~/Documents/yt-transcripts/{channel}/{title}.md for a saved video."""
        db_path = str(tmp_path / "test.duckdb")

        # Save a video so the DB has metadata to look up.
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("dQw4w9WgXcQ", transcript, _SAMPLE_METADATA)

        result = _auto_output_path("dQw4w9WgXcQ", db_path)

        assert result is not None
        # The path should end with the expected channel/title structure.
        assert result.endswith(os.path.join(
            "Rick Astley", "Never Gonna Give You Up.md",
        ))
        # The path should start with the expanded home directory.
        assert result.startswith(os.path.expanduser("~"))
        assert "yt-transcripts" in result

    def test_returns_none_for_missing_video(self, tmp_path) -> None:
        """Returns None when the video isn't in the database."""
        db_path = str(tmp_path / "test.duckdb")

        # Create an empty DB (no videos saved).
        with TranscriptStore(db_path) as _:
            pass

        result = _auto_output_path("nonexistent1", db_path)
        assert result is None

    def test_sanitizes_unsafe_characters(self, tmp_path) -> None:
        """Unsafe characters in title/channel are replaced with hyphens."""
        db_path = str(tmp_path / "test.duckdb")

        # Save a video with unsafe characters in title and channel name.
        meta = VideoMetadata(
            video_id="test1234567",
            title="What: Is This? A/B Test",
            channel_id="UC_test",
            channel_name="My <Channel>",
            channel_url=None,
            upload_date=date(2024, 1, 1),
            duration_secs=60,
        )
        with TranscriptStore(db_path) as store:
            transcript = _make_fake_transcript(_SAMPLE_SEGMENTS)
            store.save_transcript("test1234567", transcript, meta)

        result = _auto_output_path("test1234567", db_path)

        assert result is not None
        # Channel name's angle brackets should be replaced.
        assert "My -Channel-" in result
        # Title's colons, question marks, and slashes should be replaced.
        assert "What- Is This- A-B Test.md" in result


# ---------------------------------------------------------------------------
# CLI `get` subcommand — default option values
# ---------------------------------------------------------------------------

class TestGetDefaults:
    """Tests that the `get` subcommand has the right default option values."""

    @patch("yt_transcript_extractor.cli.extract")
    @patch("yt_transcript_extractor.cli._auto_output_path")
    def test_defaults_to_doc_format_and_save_true(
        self,
        mock_auto_path: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """
        Without explicit flags, `get` defaults to fmt='doc' and save=True.

        We verify this by checking the arguments passed to extract().
        """
        mock_extract.return_value = "**[00:00]** Hello world"
        # Return None so it falls back to stdout (we don't want to write files
        # in tests).
        mock_auto_path.return_value = None

        runner = CliRunner()
        result = runner.invoke(main, ["get", "dQw4w9WgXcQ"])

        assert result.exit_code == 0
        # extract() should have been called with fmt="doc" and save=True.
        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args
        assert call_kwargs[1]["fmt"] == "doc" or call_kwargs[0][2] == "doc" if len(call_kwargs[0]) > 2 else True
        # Check via keyword args — extract() is called with keyword arguments
        # in the CLI.
        _, kwargs = mock_extract.call_args
        assert kwargs.get("fmt") == "doc"
        assert kwargs.get("save") is True

    @patch("yt_transcript_extractor.cli.extract")
    def test_no_save_flag_disables_save(self, mock_extract: MagicMock) -> None:
        """The --no-save flag sets save=False, reverting to old behavior."""
        mock_extract.return_value = "plain text output"

        runner = CliRunner()
        result = runner.invoke(main, ["get", "dQw4w9WgXcQ", "--no-save", "--format", "text"])

        assert result.exit_code == 0
        _, kwargs = mock_extract.call_args
        assert kwargs.get("save") is False

    @patch("yt_transcript_extractor.cli.extract")
    def test_text_format_prints_to_stdout(self, mock_extract: MagicMock) -> None:
        """With --format text --no-save, output goes to stdout (no auto-path)."""
        mock_extract.return_value = "Hello world\nSecond line"

        runner = CliRunner()
        result = runner.invoke(main, ["get", "dQw4w9WgXcQ", "--format", "text", "--no-save"])

        assert result.exit_code == 0
        assert "Hello world" in result.output
        assert "Second line" in result.output


# ---------------------------------------------------------------------------
# CLI `get` subcommand — auto-path file writing
# ---------------------------------------------------------------------------

class TestGetAutoPath:
    """Tests for the auto-path file writing in the `get` subcommand."""

    @patch("yt_transcript_extractor.cli._auto_output_path")
    @patch("yt_transcript_extractor.cli.extract")
    def test_writes_doc_to_auto_path(
        self,
        mock_extract: MagicMock,
        mock_auto_path: MagicMock,
        tmp_path,
    ) -> None:
        """When fmt=doc and save=True, the transcript is written to auto-path."""
        mock_extract.return_value = "**[00:00]** Hello world"
        # No need to mock parse_video_id — it's a pure function that handles
        # bare 11-char IDs correctly, and "dQw4w9WgXcQ" is a valid ID.
        auto_file = str(tmp_path / "channel" / "title.md")
        mock_auto_path.return_value = auto_file

        runner = CliRunner()
        result = runner.invoke(main, ["get", "dQw4w9WgXcQ"])

        assert result.exit_code == 0
        # The file should have been created with the transcript content.
        assert os.path.exists(auto_file)
        with open(auto_file) as fh:
            content = fh.read()
        assert "**[00:00]** Hello world" in content
        # The transcript content should NOT appear in the combined output —
        # only confirmation messages (which go to stderr) should be there.
        # CliRunner mixes stderr into output, so we check that the raw
        # transcript text isn't echoed (it was written to the file instead).
        assert "**[00:00]**" not in result.output
        # Confirmation message should appear (via stderr).
        assert "Transcript written to" in result.output


# ---------------------------------------------------------------------------
# CLI `saved` subcommand — default option values
# ---------------------------------------------------------------------------

class TestSavedDefaults:
    """Tests that the `saved` subcommand has the right default option values."""

    @patch("yt_transcript_extractor.cli._auto_output_path")
    @patch("yt_transcript_extractor.cli.TranscriptStore")
    def test_defaults_to_doc_format(
        self,
        MockStore: MagicMock,
        mock_auto_path: MagicMock,
    ) -> None:
        """
        Without explicit --format, `saved` defaults to doc format.

        We verify by checking that get_transcript_doc() is called (not
        get_transcript_text() or get_transcript()).
        """
        mock_store = MagicMock()
        mock_store.has_video.return_value = True
        mock_store.get_transcript_doc.return_value = "**[00:00]** Hello"
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)
        # Return None so it falls back to stdout.
        mock_auto_path.return_value = None

        runner = CliRunner()
        result = runner.invoke(main, ["saved", "dQw4w9WgXcQ"])

        assert result.exit_code == 0
        # get_transcript_doc() should have been called (doc is the default).
        mock_store.get_transcript_doc.assert_called_once_with("dQw4w9WgXcQ")
        # get_transcript_text() should NOT have been called.
        mock_store.get_transcript_text.assert_not_called()


# ---------------------------------------------------------------------------
# CLI `saved` subcommand — auto-path file writing
# ---------------------------------------------------------------------------

class TestSavedAutoPath:
    """Tests for the auto-path file writing in the `saved` subcommand."""

    @patch("yt_transcript_extractor.cli._auto_output_path")
    @patch("yt_transcript_extractor.cli.TranscriptStore")
    def test_writes_doc_to_auto_path(
        self,
        MockStore: MagicMock,
        mock_auto_path: MagicMock,
        tmp_path,
    ) -> None:
        """When fmt=doc (default), the transcript is written to auto-path."""
        mock_store = MagicMock()
        mock_store.has_video.return_value = True
        mock_store.get_transcript_doc.return_value = "**[00:00]** Saved content"
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        auto_file = str(tmp_path / "channel" / "title.md")
        mock_auto_path.return_value = auto_file

        runner = CliRunner()
        result = runner.invoke(main, ["saved", "dQw4w9WgXcQ"])

        assert result.exit_code == 0
        # The file should have been created.
        assert os.path.exists(auto_file)
        with open(auto_file) as fh:
            content = fh.read()
        assert "**[00:00]** Saved content" in content
        # The transcript content should NOT appear in the combined output —
        # only confirmation messages (which go to stderr) should be there.
        assert "**[00:00]**" not in result.output
        # Confirmation message should appear (via stderr).
        assert "Transcript written to" in result.output

    @patch("yt_transcript_extractor.cli.TranscriptStore")
    def test_text_format_prints_to_stdout(self, MockStore: MagicMock) -> None:
        """With --format text, output goes to stdout (no auto-path)."""
        mock_store = MagicMock()
        mock_store.has_video.return_value = True
        mock_store.get_transcript_text.return_value = "Hello\nWorld"
        MockStore.return_value.__enter__ = MagicMock(return_value=mock_store)
        MockStore.return_value.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        result = runner.invoke(main, ["saved", "dQw4w9WgXcQ", "--format", "text"])

        assert result.exit_code == 0
        assert "Hello" in result.output
        assert "World" in result.output
