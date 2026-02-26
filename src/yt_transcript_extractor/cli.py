"""
cli.py — Command-line interface for yt-transcript-extractor.

Provides the `yt-transcript` command group (registered as a console script
in pyproject.toml).  The CLI is organized into subcommands:

    get       Fetch a transcript from YouTube (optionally saving to DB).
    channels  List all channels with saved transcripts.
    videos    List saved videos for a specific channel.
    saved     Retrieve a previously saved transcript from the local DB.
    search    Search across all saved transcripts for a keyword/phrase.

Usage examples:
    yt-transcript get "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    yt-transcript get dQw4w9WgXcQ --save --format json
    yt-transcript channels
    yt-transcript videos UC38IQsAvIsxxjztdMZQtwHA
    yt-transcript saved dQw4w9WgXcQ
    yt-transcript search "never gonna give you up"
"""

from __future__ import annotations

import json
import os
import re
import sys

import click

from yt_transcript_extractor.errors import TranscriptError
from yt_transcript_extractor.extractor import extract
from yt_transcript_extractor.storage import TranscriptStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default database path used by all subcommands.  Can be overridden with --db.
_DEFAULT_DB = "transcripts.duckdb"

# Base directory for auto-saved transcript documents.  Expands ~ at runtime
# so it works on any user's machine.
_AUTO_OUTPUT_BASE = os.path.join("~", "Documents", "yt-transcripts")

# Characters that are unsafe in filenames on Windows and/or POSIX systems.
# We replace these with a hyphen when building auto-output paths.
_UNSAFE_FILENAME_CHARS = re.compile(r'[:/\\?*<>|"]')


# ---------------------------------------------------------------------------
# Helper functions — filename sanitization and auto-output path generation
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """
    Replace filesystem-unsafe characters with hyphens and clean up whitespace.

    Takes a raw string (e.g. a video title or channel name) and returns a
    version that's safe to use as a filename on both Windows and POSIX.
    Specifically:
      - Replaces : / \\ ? * < > | " with hyphens.
      - Strips leading/trailing whitespace and dots (dots at the start
        create hidden files on POSIX; dots at the end cause issues on Windows).

    Args:
        name: The raw string to sanitize (e.g. "My Video: Part 1/2").

    Returns:
        A sanitized string safe for use as a filename (e.g. "My Video- Part 1-2").
    """
    sanitized = _UNSAFE_FILENAME_CHARS.sub("-", name)
    return sanitized.strip().strip(".")


def _auto_output_path(video_id: str, db: str) -> str | None:
    """
    Build an automatic output file path for a video's transcript document.

    Looks up the video's title and channel name from the local database and
    constructs a path like:
        ~/Documents/yt-transcripts/{sanitized_channel}/{sanitized_title}.md

    This is the "just works" path — the user runs `yt-transcript get <url>`
    and gets a nicely organized markdown file without specifying --output.

    Args:
        video_id: The 11-character YouTube video ID to look up.
        db:       Path to the DuckDB database file containing video metadata.

    Returns:
        The full expanded file path as a string, or None if the video isn't
        in the database (which means we can't determine channel/title).
    """
    try:
        with TranscriptStore(db) as store:
            if not store.has_video(video_id):
                return None

            # Query the video's title and its channel name by joining
            # the videos and channels tables.
            row = store.conn.execute(
                """
                SELECT v.title, c.channel_name
                FROM videos v
                JOIN channels c ON v.channel_id = c.channel_id
                WHERE v.video_id = ?
                """,
                [video_id],
            ).fetchone()

            if row is None:
                return None

            title, channel_name = row[0], row[1]
    except TranscriptError:
        # If the DB can't be opened (e.g. first run, corrupt file), we
        # can't determine the path — fall back to stdout.
        return None

    # Build the path: ~/Documents/yt-transcripts/{channel}/{title}.md
    sanitized_channel = _sanitize_filename(channel_name)
    sanitized_title = _sanitize_filename(title)
    base = os.path.expanduser(_AUTO_OUTPUT_BASE)
    return os.path.join(base, sanitized_channel, f"{sanitized_title}.md")


# ---------------------------------------------------------------------------
# CLI group — the top-level `yt-transcript` command
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """
    YouTube Transcript Extractor — fetch, save, and search video transcripts.
    """
    # The group itself does nothing; each subcommand handles its own logic.
    pass


# ---------------------------------------------------------------------------
# Subcommand: get — fetch a transcript from YouTube
# ---------------------------------------------------------------------------

@main.command()
@click.argument("video", metavar="URL_OR_ID")
@click.option(
    "--format", "-f",
    "fmt",                           # avoid shadowing the builtin "format"
    type=click.Choice(["text", "json", "doc"], case_sensitive=False),
    default="doc",
    show_default=True,
    help="Output format: plain text, JSON with timestamps, or readable markdown document.",
)
@click.option(
    "--lang", "-l",
    default=None,
    help="Comma-separated language codes in priority order (e.g. 'de,en'). Defaults to English.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Write output to a file instead of stdout.",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Save the transcript to a local DuckDB database for offline access.",
)
@click.option(
    "--db",
    default=_DEFAULT_DB,
    show_default=True,
    help="Path to the DuckDB database file (only used with --save).",
)
def get(
    video: str,
    fmt: str,
    lang: str | None,
    output: str | None,
    save: bool,
    db: str,
) -> None:
    """
    Fetch a YouTube video transcript.

    VIDEO can be a full YouTube URL or an 11-character video ID.
    By default, saves to the local DB and writes a markdown document to
    ~/Documents/yt-transcripts/{channel}/{title}.md.
    Use --no-save to skip DB persistence, or --format text/json for stdout.
    """
    # Parse the comma-separated language list into a proper list, if provided.
    languages: list[str] | None = None
    if lang:
        languages = [code.strip() for code in lang.split(",")]

    try:
        result = extract(
            video,
            languages=languages,
            fmt=fmt,
            save=save,
            db_path=db if save else None,
        )
    except TranscriptError as exc:
        # Print a clean, human-readable error message to stderr and exit with
        # a non-zero code.  We don't dump a traceback — it's not helpful for
        # end-users and the exception message already says what went wrong.
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)

    if save:
        click.echo(f"Transcript saved to {db}", err=True)

    # Serialise dict output to a JSON string for display / file writing.
    if isinstance(result, dict):
        text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        text = result

    # Write to file or stdout.
    if output:
        # Explicit --output given — write to that exact path.
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
        click.echo(f"Transcript written to {output}", err=True)
    elif fmt == "doc" and save:
        # Auto-path mode: doc format + save is on + no explicit --output.
        # We need the video_id to look up metadata.  extract() was called with
        # the raw URL/ID, so we parse it again to get the canonical 11-char ID.
        from yt_transcript_extractor.extractor import parse_video_id
        video_id = parse_video_id(video)
        auto_path = _auto_output_path(video_id, db)

        if auto_path:
            # Create the directory tree (e.g. ~/Documents/yt-transcripts/Channel/)
            # if it doesn't exist yet.
            os.makedirs(os.path.dirname(auto_path), exist_ok=True)
            with open(auto_path, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.write("\n")
            click.echo(f"Transcript written to {auto_path}", err=True)
        else:
            # Fallback: couldn't determine path (shouldn't happen since save
            # succeeded, but be safe) — print to stdout.
            click.echo(text)
    else:
        # Non-doc format or save is off — print to stdout as before.
        click.echo(text)


# ---------------------------------------------------------------------------
# Subcommand: channels — list all channels with saved transcripts
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--db",
    default=_DEFAULT_DB,
    show_default=True,
    help="Path to the DuckDB database file.",
)
def channels(db: str) -> None:
    """
    List all channels that have saved transcripts.

    Shows each channel's name, ID, and the number of saved videos.
    """
    try:
        with TranscriptStore(db) as store:
            channel_list = store.list_channels()
    except TranscriptError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)

    if not channel_list:
        click.echo("No saved channels found. Use 'yt-transcript get --save' to save transcripts.")
        return

    # Display channels in a readable format with video counts.
    for ch in channel_list:
        video_word = "video" if ch.video_count == 1 else "videos"
        click.echo(f"{ch.channel_name} ({ch.video_count} {video_word})")
        click.echo(f"  ID: {ch.channel_id}")
        if ch.channel_url:
            click.echo(f"  URL: {ch.channel_url}")
        click.echo()


# ---------------------------------------------------------------------------
# Subcommand: videos — list saved videos for a channel
# ---------------------------------------------------------------------------

@main.command()
@click.argument("channel_id")
@click.option(
    "--db",
    default=_DEFAULT_DB,
    show_default=True,
    help="Path to the DuckDB database file.",
)
def videos(channel_id: str, db: str) -> None:
    """
    List all saved videos for a specific channel.

    CHANNEL_ID is the YouTube channel identifier (e.g. UC38IQsAvIsxxjztdMZQtwHA).
    Use 'yt-transcript channels' to find channel IDs.
    """
    try:
        with TranscriptStore(db) as store:
            video_list = store.list_videos(channel_id)
    except TranscriptError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)

    if not video_list:
        click.echo(f"No saved videos found for channel {channel_id}.")
        return

    # Display each video with its title, ID, and upload date.
    for v in video_list:
        date_str = str(v.upload_date) if v.upload_date else "unknown date"
        click.echo(f"[{date_str}] {v.title}")
        click.echo(f"  ID: {v.video_id}")
        click.echo()


# ---------------------------------------------------------------------------
# Subcommand: saved — retrieve a stored transcript
# ---------------------------------------------------------------------------

@main.command()
@click.argument("video_id")
@click.option(
    "--format", "-f",
    "fmt",
    type=click.Choice(["text", "json", "doc"], case_sensitive=False),
    default="doc",
    show_default=True,
    help="Output format: plain text, JSON with timestamps, or readable markdown document.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Write output to a file instead of stdout.",
)
@click.option(
    "--db",
    default=_DEFAULT_DB,
    show_default=True,
    help="Path to the DuckDB database file.",
)
def saved(video_id: str, fmt: str, output: str | None, db: str) -> None:
    """
    Retrieve a previously saved transcript from the local database.

    VIDEO_ID is the 11-character YouTube video identifier.
    This does NOT fetch from YouTube — it only reads from the local DB.
    By default, writes a markdown document to
    ~/Documents/yt-transcripts/{channel}/{title}.md.
    """
    try:
        with TranscriptStore(db) as store:
            if not store.has_video(video_id):
                click.echo(
                    f"Error: Video {video_id} not found in database. "
                    f"Use 'yt-transcript get {video_id}' to save it first.",
                    err=True,
                )
                sys.exit(1)

            if fmt == "json":
                segments = store.get_transcript(video_id)
                result: str | dict = {
                    "video_id": video_id,
                    "segment_count": len(segments),
                    "segments": segments,
                }
            elif fmt == "doc":
                result = store.get_transcript_doc(video_id)
            else:
                result = store.get_transcript_text(video_id)
    except TranscriptError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)

    # Serialise dict output to a JSON string for display / file writing.
    if isinstance(result, dict):
        text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        text = result

    # Write to file or stdout.
    if output:
        # Explicit --output given — write to that exact path.
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
        click.echo(f"Transcript written to {output}", err=True)
    elif fmt == "doc":
        # Auto-path mode: doc format + no explicit --output.
        # Build the path from DB metadata and write the file there.
        auto_path = _auto_output_path(video_id, db)

        if auto_path:
            os.makedirs(os.path.dirname(auto_path), exist_ok=True)
            with open(auto_path, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.write("\n")
            click.echo(f"Transcript written to {auto_path}", err=True)
        else:
            # Fallback: couldn't determine path — print to stdout.
            click.echo(text)
    else:
        # Non-doc format — print to stdout as before.
        click.echo(text)


# ---------------------------------------------------------------------------
# Subcommand: search — full-text search across saved transcripts
# ---------------------------------------------------------------------------

@main.command()
@click.argument("query")
@click.option(
    "--db",
    default=_DEFAULT_DB,
    show_default=True,
    help="Path to the DuckDB database file.",
)
def search(query: str, db: str) -> None:
    """
    Search across all saved transcripts for a keyword or phrase.

    QUERY is a case-insensitive substring to search for in transcript text.
    Results show matching segments with their video context and timestamps.
    """
    try:
        with TranscriptStore(db) as store:
            results = store.search_transcripts(query)
    except TranscriptError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)

    if not results:
        click.echo(f"No results found for '{query}'.")
        return

    # Group results by video for readable output.
    current_video = None
    for r in results:
        # Print a header when we move to a new video.
        if r["video_id"] != current_video:
            current_video = r["video_id"]
            click.echo(f"\n{r['title']} ({r['channel_name']})")
            click.echo(f"  Video ID: {r['video_id']}")

        # Format the timestamp as MM:SS for readability.
        total_secs = int(r["start"])
        minutes, seconds = divmod(total_secs, 60)
        click.echo(f"  [{minutes:02d}:{seconds:02d}] {r['text']}")
