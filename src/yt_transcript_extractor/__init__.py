"""
yt_transcript_extractor — Extract YouTube video transcripts.

Public API:
    extract()               High-level one-call interface (URL → formatted output).
    get_transcript()        Fetch raw transcript segments for a video ID.
    parse_video_id()        Parse a YouTube URL or validate a bare video ID.
    fetch_video_metadata()  Fetch video/channel metadata via yt-dlp.
    TranscriptStore         DuckDB-backed storage for transcripts.
    VideoMetadata           Dataclass holding video metadata fields.

Exception hierarchy (all importable from this package):
    TranscriptError              Base exception for all transcript errors.
    ├── VideoNotFoundError       Video ID doesn't exist or is private.
    ├── TranscriptUnavailableError  Video exists but has no transcripts.
    ├── LanguageNotAvailableError   Requested language not available.
    ├── MetadataFetchError       yt-dlp metadata fetch failed.
    └── StorageError             DuckDB operation failed.

Usage:
    from yt_transcript_extractor import extract
    text = extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # With saving to local database:
    text = extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ", save=True)
"""

from yt_transcript_extractor.extractor import (
    extract,
    get_transcript,
    parse_video_id,
)
from yt_transcript_extractor.errors import (
    LanguageNotAvailableError,
    MetadataFetchError,
    StorageError,
    TranscriptError,
    TranscriptUnavailableError,
    VideoNotFoundError,
)
from yt_transcript_extractor.metadata import (
    VideoMetadata,
    fetch_video_metadata,
)
from yt_transcript_extractor.storage import TranscriptStore

__all__ = [
    "extract",
    "get_transcript",
    "parse_video_id",
    "fetch_video_metadata",
    "TranscriptStore",
    "VideoMetadata",
    "TranscriptError",
    "VideoNotFoundError",
    "TranscriptUnavailableError",
    "LanguageNotAvailableError",
    "MetadataFetchError",
    "StorageError",
]
