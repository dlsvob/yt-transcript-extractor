"""
extractor.py — Core transcript extraction logic.

This is the heart of yt-transcript-extractor.  It wraps the
`youtube-transcript-api` library and exposes a clean, high-level interface for:

    1. Parsing YouTube URLs / IDs  → parse_video_id()
    2. Fetching raw transcript data → get_transcript()
    3. Formatting output            → format_text(), format_json(), format_doc()
    4. One-call convenience         → extract()

Only single-video extraction is supported (no playlists).
"""

from __future__ import annotations

import json
import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._transcripts import FetchedTranscript
import youtube_transcript_api as yta_errors  # exception classes live here

from yt_transcript_extractor.errors import (
    LanguageNotAvailableError,
    TranscriptError,
    TranscriptUnavailableError,
    VideoNotFoundError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex patterns that cover the most common YouTube URL shapes:
#   - https://www.youtube.com/watch?v=VIDEO_ID
#   - https://youtu.be/VIDEO_ID
#   - https://www.youtube.com/embed/VIDEO_ID
#   - https://www.youtube.com/shorts/VIDEO_ID
#   - https://www.youtube.com/v/VIDEO_ID
# Each pattern captures the 11-character video ID in group "id".
_URL_PATTERNS: list[re.Pattern[str]] = [
    # Standard watch URL — the ID sits in the "v" query parameter
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=(?P<id>[A-Za-z0-9_-]{11})"),
    # Short share URL — ID is the path segment right after the domain
    re.compile(r"(?:https?://)?youtu\.be/(?P<id>[A-Za-z0-9_-]{11})"),
    # Embed / shorts / old "v/" URLs — ID follows the path prefix
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/(?:embed|shorts|v)/(?P<id>[A-Za-z0-9_-]{11})"),
]

# A bare video ID is exactly 11 characters from the base64url alphabet.
_BARE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Default language fallback order: English first, then auto-generated.
# youtube-transcript-api tries languages in order and falls back automatically,
# but we add "en" up front so English is preferred when no language is specified.
_DEFAULT_LANGUAGES = ["en"]


# ---------------------------------------------------------------------------
# URL / ID parsing
# ---------------------------------------------------------------------------

def parse_video_id(url_or_id: str) -> str:
    """
    Extract a YouTube video ID from a URL string, or validate a raw 11-char ID.

    Accepts all common YouTube URL formats (watch, short, embed, shorts) as
    well as a bare 11-character ID string.

    Args:
        url_or_id: A YouTube URL or a raw video ID.

    Returns:
        The 11-character video ID.

    Raises:
        VideoNotFoundError: If the string doesn't match any known format.
    """
    url_or_id = url_or_id.strip()

    # Try each URL pattern first — order doesn't matter since they're disjoint.
    for pattern in _URL_PATTERNS:
        match = pattern.search(url_or_id)
        if match:
            return match.group("id")

    # Fall back to treating the input as a bare ID.
    if _BARE_ID_PATTERN.match(url_or_id):
        return url_or_id

    # Nothing matched — the input isn't a recognisable YouTube reference.
    raise VideoNotFoundError(url_or_id)


# ---------------------------------------------------------------------------
# Transcript fetching
# ---------------------------------------------------------------------------

def get_transcript(
    video_id: str,
    languages: list[str] | None = None,
) -> FetchedTranscript:
    """
    Fetch transcript segments for a single YouTube video.

    Uses youtube-transcript-api under the hood.  Language fallback logic:
    if specific languages are requested, try those first; otherwise fall back
    to English → auto-generated.

    Args:
        video_id:  The 11-character YouTube video ID (NOT a full URL).
        languages: Optional list of language codes in descending priority
                   (e.g. ["de", "en"]).  When None, defaults to ["en"].

    Returns:
        A FetchedTranscript object (iterable of snippets with .text, .start,
        .duration attributes).

    Raises:
        VideoNotFoundError:          The video ID doesn't exist or is private.
        TranscriptUnavailableError:  The video exists but has no transcripts.
        LanguageNotAvailableError:   Transcripts exist, but not in the
                                     requested language(s).
        TranscriptError:             Catch-all for unexpected upstream errors.
    """
    langs = languages if languages else _DEFAULT_LANGUAGES

    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=langs)
        return transcript

    # --- Map upstream exceptions to our own hierarchy ---
    except yta_errors.InvalidVideoId:
        raise VideoNotFoundError(video_id)
    except yta_errors.VideoUnavailable:
        raise VideoNotFoundError(video_id)
    except yta_errors.TranscriptsDisabled:
        raise TranscriptUnavailableError(video_id)
    except yta_errors.NoTranscriptFound:
        # Transcripts exist, but not in the requested language.
        raise LanguageNotAvailableError(video_id, langs)
    except yta_errors.CouldNotRetrieveTranscript as exc:
        # Generic upstream error — wrap it so callers only need to catch
        # our TranscriptError hierarchy.
        raise TranscriptError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_text(transcript: FetchedTranscript) -> str:
    """
    Convert transcript segments into plain text, one line per segment.

    This is the simplest output format — just the spoken words, no timestamps.
    Useful for feeding into summarisers, search indexes, or reading directly.

    Args:
        transcript: A FetchedTranscript (iterable of snippets).

    Returns:
        A single string with one transcript line per text line.
    """
    return "\n".join(snippet.text for snippet in transcript)


def format_json(transcript: FetchedTranscript, video_id: str) -> dict:
    """
    Build a structured JSON-serialisable dict from transcript segments.

    Includes video metadata alongside the full list of timestamped segments,
    making it easy to store or post-process programmatically.

    Args:
        transcript: A FetchedTranscript (iterable of snippets).
        video_id:   The video ID (included in the output for traceability).

    Returns:
        A dict with keys: video_id, segment_count, segments.
        Each segment has: text, start, duration.
    """
    segments = transcript.to_raw_data()
    return {
        "video_id": video_id,
        "segment_count": len(segments),
        "segments": segments,
    }


# Paragraph boundary interval for the "doc" format.  Segments are grouped
# into flowing paragraphs; a new paragraph starts whenever the segment's
# start time crosses the next multiple of this threshold (in seconds).
_DOC_PARAGRAPH_INTERVAL_SECS = 30


def _seconds_to_mmss(seconds: float) -> str:
    """
    Convert a float timestamp (in seconds) to a MM:SS string.

    Used by format_doc() to produce human-readable time markers.
    Values above 59:59 wrap naturally (e.g. 3661.0 → "61:01").

    Args:
        seconds: Timestamp in seconds (e.g. 92.5).

    Returns:
        A string like "01:32".
    """
    total = int(seconds)
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def format_doc(transcript) -> str:
    """
    Convert transcript segments into a readable markdown document.

    Segments are joined with spaces into flowing paragraphs, with a new
    paragraph starting every ~30 seconds.  Each paragraph is prefixed with
    a bold **[MM:SS]** timestamp marking the start of that time window.
    Paragraphs are separated by blank lines for clean markdown rendering.

    This format is designed to be human-readable and converts well to docx
    via pandoc or similar tools.

    The transcript argument can be either a FetchedTranscript (iterable of
    snippet objects with .text and .start attrs) or a list of dicts with
    "text" and "start" keys — allowing reuse from both the live-fetch and
    stored-segment code paths.

    Args:
        transcript: A FetchedTranscript or list of {"text", "start", ...} dicts.

    Returns:
        A markdown string with timestamped paragraphs.  Returns an empty
        string if the transcript has no segments.
    """
    paragraphs: list[str] = []
    current_texts: list[str] = []
    # Track which 30-second bucket the current paragraph belongs to.
    # None means we haven't started yet.
    paragraph_start: float | None = None

    for snippet in transcript:
        # Support both FetchedTranscript snippet objects (.start, .text)
        # and plain dicts from stored segments ({"start": ..., "text": ...}).
        if isinstance(snippet, dict):
            start = snippet["start"]
            text = snippet["text"]
        else:
            start = snippet.start
            text = snippet.text

        # Decide whether this segment starts a new paragraph.  A new
        # paragraph begins when (a) it's the very first segment, or
        # (b) the segment's start time has crossed into the next
        # 30-second bucket.
        if paragraph_start is None:
            # Very first segment — begin the first paragraph.
            paragraph_start = start
            current_texts.append(text)
        elif start - paragraph_start >= _DOC_PARAGRAPH_INTERVAL_SECS:
            # Time threshold crossed — flush the current paragraph and
            # start a new one.
            timestamp = _seconds_to_mmss(paragraph_start)
            paragraphs.append(f"**[{timestamp}]** {' '.join(current_texts)}")
            paragraph_start = start
            current_texts = [text]
        else:
            # Still within the same time bucket — append to current paragraph.
            current_texts.append(text)

    # Flush the last paragraph (if any segments existed).
    if current_texts and paragraph_start is not None:
        timestamp = _seconds_to_mmss(paragraph_start)
        paragraphs.append(f"**[{timestamp}]** {' '.join(current_texts)}")

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# High-level convenience function (main public API)
# ---------------------------------------------------------------------------

def extract(
    url_or_id: str,
    languages: list[str] | None = None,
    fmt: str = "text",
    *,
    save: bool = False,
    db_path: str | None = None,
) -> str | dict:
    """
    One-call interface: parse URL → fetch transcript → format output.

    This is the recommended entry point for most users.  It chains
    parse_video_id → get_transcript → format_text / format_json / format_doc.

    When save=True, the transcript and video metadata are also persisted
    to a DuckDB database for offline retrieval and search.  Metadata
    (channel name, video title, upload date) is fetched via yt-dlp.

    Args:
        url_or_id:  A YouTube URL or raw video ID.
        languages:  Optional language priority list (e.g. ["de", "en"]).
        fmt:        Output format — "text" for plain text, "json" for a dict
                    with timestamps, "doc" for a readable markdown document
                    with timestamped paragraphs.
        save:       If True, persist the transcript to DuckDB.  Requires
                    yt-dlp and duckdb (both are installed dependencies).
        db_path:    Path to the DuckDB file.  Defaults to "transcripts.duckdb"
                    in the current working directory.  Only used when save=True.

    Returns:
        A plain-text string (fmt="text"), a dict (fmt="json"), or a markdown
        string (fmt="doc").

    Raises:
        ValueError:          If fmt is not "text", "json", or "doc".
        TranscriptError:     (or subclass) on any extraction failure.
        MetadataFetchError:  If save=True and yt-dlp can't fetch metadata.
        StorageError:        If save=True and the database operation fails.
    """
    if fmt not in ("text", "json", "doc"):
        raise ValueError(f"Unknown format {fmt!r}; expected 'text', 'json', or 'doc'")

    video_id = parse_video_id(url_or_id)
    transcript = get_transcript(video_id, languages=languages)

    # Optionally persist the transcript to DuckDB.  Imports are inside the
    # if-block so the default (save=False) path has zero overhead from yt-dlp
    # or duckdb imports.  FetchedTranscript is safely re-iterable (backed by
    # a list internally), so saving first then formatting works correctly.
    if save:
        from yt_transcript_extractor.metadata import fetch_video_metadata
        from yt_transcript_extractor.storage import TranscriptStore

        metadata = fetch_video_metadata(video_id)
        store_path = db_path or "transcripts.duckdb"
        with TranscriptStore(store_path) as store:
            store.save_transcript(video_id, transcript, metadata)

    if fmt == "json":
        return format_json(transcript, video_id)

    if fmt == "doc":
        return format_doc(transcript)

    return format_text(transcript)
