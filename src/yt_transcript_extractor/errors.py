"""
errors.py — Custom exception hierarchy for yt-transcript-extractor.

Every exception carries an `http_status` attribute so the FastAPI error
handler can translate library-level errors directly into the correct HTTP
response code without a separate mapping table.

Hierarchy:
    TranscriptError (base, 500)
    ├── VideoNotFoundError (404)
    ├── TranscriptUnavailableError (404)
    ├── LanguageNotAvailableError (400)
    ├── MetadataFetchError (502)
    └── StorageError (500)
"""


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class TranscriptError(Exception):
    """
    Root exception for all transcript-related errors.

    Attributes:
        message:     Human-readable description of what went wrong.
        http_status: Suggested HTTP status code for the API layer.
    """

    def __init__(self, message: str, http_status: int = 500) -> None:
        super().__init__(message)
        self.message = message
        # Why store status on the exception?  It lets the FastAPI error handler
        # use a single generic handler instead of one handler per exception type.
        self.http_status = http_status


# ---------------------------------------------------------------------------
# Specific error cases
# ---------------------------------------------------------------------------

class VideoNotFoundError(TranscriptError):
    """
    Raised when the video ID doesn't correspond to a real YouTube video.

    Possible causes: typo in the ID, video was deleted, or video is private.
    Maps to HTTP 404.
    """

    def __init__(self, video_id: str) -> None:
        super().__init__(
            message=f"Video not found: {video_id}",
            http_status=404,
        )
        self.video_id = video_id


class TranscriptUnavailableError(TranscriptError):
    """
    Raised when the video exists but has no transcript at all.

    This happens for videos where the creator disabled captions and YouTube
    hasn't generated automatic ones (e.g. music-only content, very new uploads).
    Maps to HTTP 404.
    """

    def __init__(self, video_id: str) -> None:
        super().__init__(
            message=f"No transcript available for video: {video_id}",
            http_status=404,
        )
        self.video_id = video_id


class LanguageNotAvailableError(TranscriptError):
    """
    Raised when the video has transcripts, but none in the requested language.

    The caller asked for a specific language (e.g. "fr") but only other
    languages are available.  Maps to HTTP 400 because it's a client-side
    request issue — the resource exists, just not in that language.
    """

    def __init__(self, video_id: str, requested: list[str]) -> None:
        langs = ", ".join(requested)
        super().__init__(
            message=f"Transcript not available in language(s) [{langs}] for video: {video_id}",
            http_status=400,
        )
        self.video_id = video_id
        self.requested = requested


class MetadataFetchError(TranscriptError):
    """
    Raised when yt-dlp fails to retrieve video metadata from YouTube.

    This typically means a network issue, rate-limiting, or a transient
    YouTube outage.  Maps to HTTP 502 because the failure is upstream —
    our server is healthy but the third-party source returned an error.
    """

    def __init__(self, video_id: str, reason: str = "") -> None:
        detail = f": {reason}" if reason else ""
        super().__init__(
            message=f"Failed to fetch metadata for video {video_id}{detail}",
            http_status=502,
        )
        self.video_id = video_id


class StorageError(TranscriptError):
    """
    Raised when a DuckDB database operation fails unexpectedly.

    Covers issues like file permission errors, corrupt database files, or
    schema migration problems.  Maps to HTTP 500 because it's an internal
    server issue — the caller's request was valid but our storage layer broke.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            http_status=500,
        )
