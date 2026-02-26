"""
metadata.py — Fetch video metadata from YouTube using yt-dlp.

The youtube-transcript-api library gives us transcript segments but no
channel information (channel name, channel ID, video title, upload date).
This module fills that gap by using yt-dlp in metadata-only mode — it
extracts structured info from the YouTube page without downloading any
audio or video content.

The main entry point is fetch_video_metadata(), which returns a
VideoMetadata dataclass containing everything we need to organize
transcripts by channel in the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import yt_dlp

from yt_transcript_extractor.errors import MetadataFetchError


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoMetadata:
    """
    Structured metadata for a single YouTube video.

    All fields come from yt-dlp's info_dict.  frozen=True makes instances
    hashable and prevents accidental mutation after creation.

    Attributes:
        video_id:     The 11-character YouTube video identifier.
        title:        The video title as displayed on YouTube.
        channel_id:   YouTube's internal channel identifier (e.g. "UC...").
        channel_name: The human-readable channel name.
        channel_url:  Full URL to the channel page.
        upload_date:  The date the video was published (may be None for
                      livestreams or unlisted videos with no date).
        duration_secs: Video length in seconds (may be None for livestreams).
    """
    video_id: str
    title: str
    channel_id: str
    channel_name: str
    channel_url: str | None
    upload_date: date | None
    duration_secs: int | None


# ---------------------------------------------------------------------------
# Metadata fetching
# ---------------------------------------------------------------------------

def fetch_video_metadata(video_id: str) -> VideoMetadata:
    """
    Fetch metadata for a YouTube video without downloading the video itself.

    Uses yt-dlp in "skip_download" mode — it only fetches the page metadata
    (title, channel, upload date, duration) and does not download any media.
    This is fast (typically < 1 second) and low-bandwidth.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        A VideoMetadata dataclass with all available metadata fields.

    Raises:
        MetadataFetchError: If yt-dlp can't retrieve the video info
            (e.g. video doesn't exist, is private, or network error).
    """
    # yt-dlp options: skip_download avoids downloading media; quiet and
    # no_warnings suppress console output since we only want the info_dict.
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        # yt-dlp raises DownloadError for any extraction failure — missing
        # video, geo-restriction, rate-limiting, etc.  We wrap it in our
        # own exception so callers only need to handle our error hierarchy.
        raise MetadataFetchError(video_id, reason=str(exc)) from exc

    # Shouldn't happen if extract_info succeeded, but guard against it.
    if info is None:
        raise MetadataFetchError(video_id, reason="yt-dlp returned no info")

    # Parse the upload date from yt-dlp's YYYYMMDD string format into a
    # proper date object.  Some videos (e.g. ongoing livestreams) may not
    # have an upload_date, so we handle None gracefully.
    raw_date = info.get("upload_date")
    upload_date: date | None = None
    if raw_date:
        try:
            upload_date = date(
                year=int(raw_date[:4]),
                month=int(raw_date[4:6]),
                day=int(raw_date[6:8]),
            )
        except (ValueError, IndexError):
            # Malformed date string — not critical, just leave it as None.
            upload_date = None

    return VideoMetadata(
        video_id=video_id,
        title=info.get("title", "Unknown Title"),
        channel_id=info.get("channel_id", ""),
        channel_name=info.get("channel", info.get("uploader", "Unknown Channel")),
        channel_url=info.get("channel_url"),
        upload_date=upload_date,
        duration_secs=info.get("duration"),
    )
