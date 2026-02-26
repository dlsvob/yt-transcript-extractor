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
    default="text",
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
    "--save", "-s",
    is_flag=True,
    default=False,
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
    Use --save to persist the transcript locally for later retrieval.
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
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
        click.echo(f"Transcript written to {output}", err=True)
    else:
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
    default="text",
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
    """
    try:
        with TranscriptStore(db) as store:
            if not store.has_video(video_id):
                click.echo(
                    f"Error: Video {video_id} not found in database. "
                    f"Use 'yt-transcript get --save {video_id}' to save it first.",
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
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
        click.echo(f"Transcript written to {output}", err=True)
    else:
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
