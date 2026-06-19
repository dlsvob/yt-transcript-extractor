"""
cli.py — Command-line interface for yt-transcript-extractor.

Provides the `yt-transcript` command group (registered as a console script
in pyproject.toml).  The CLI is organized into subcommands:

    get        Fetch transcript(s) from YouTube (optionally saving to DB).
    channels   List all channels with saved transcripts.
    videos     List saved videos for a specific channel.
    saved      Retrieve previously saved transcript(s) from the local DB.
    search     Search across all saved transcripts for a keyword/phrase.
    yt-search  Search YouTube for videos and optionally generate tutorials.

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
import concurrent.futures
import socket
import subprocess
import sys

import click

from yt_transcript_extractor.errors import TranscriptError
from yt_transcript_extractor.extractor import extract, format_tutorial
from yt_transcript_extractor.search import (
    PERIOD_MAP,
    search_youtube,
    format_duration,
    format_view_count,
)
from yt_transcript_extractor.storage import TranscriptStore
from yt_transcript_extractor.tutorials import TUTORIAL_STYLES


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
        ~/Documents/yt-transcripts/{sanitized_channel}/{sanitized_title}.html

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

    # Build the path: ~/Documents/yt-transcripts/{channel}/{title}.html
    sanitized_channel = _sanitize_filename(channel_name)
    sanitized_title = _sanitize_filename(title)
    base = os.path.expanduser(_AUTO_OUTPUT_BASE)
    return os.path.join(base, sanitized_channel, f"{sanitized_title}.html")


# ---------------------------------------------------------------------------
# CLI group — the top-level `yt-transcript` command
# ---------------------------------------------------------------------------


def _discover_ollama() -> str | None:
    """
    Scan the local network for an Ollama server on port 11434.
    Returns the first discovered URL (e.g. http://192.168.1.215:11434) or None.
    """
    import socket
    
    def check_ip(ip: str) -> str | None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((ip, 11434)) == 0:
                    return f"http://{ip}:11434"
        except Exception:
            pass
        return None

    # Get local IP to guess subnet
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        prefix = ".".join(local_ip.split(".")[:-1])
        ips = [f"{prefix}.{i}" for i in range(1, 255)]
        
        click.echo(f"Discovering Ollama servers on {prefix}.0/24...", err=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(check_ip, ip) for ip in ips]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    # Clean up other futures
                    executor.shutdown(wait=False, cancel_futures=True)
                    return result
    except Exception as exc:
        click.echo(f"Discovery error: {exc}", err=True)
        
    return None


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
@click.argument("videos", metavar="URL_OR_ID...", nargs=-1, required=True)
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
@click.option(
    "--open/--no-open",
    "open_browser",
    default=False,
    help="Open the generated HTML file in Chromium.",
)
@click.option(
    "--tutor/--no-tutor",
    default=False,
    help="Generate a tutorial using rlm-based-tutor.",
)
@click.option(
    "--tutor-endpoint",
    type=click.Choice(["auto", "anthropic", "ollama"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="LLM endpoint for tutor generation.",
)
@click.option(
    "--tutor-ollama-url",
    default=None,
    help="Custom Ollama server URL for tutorial generation.",
)
@click.option(
    "--tutor-discover/--no-tutor-discover",
    default=False,
    help="Automatically discover Ollama server on local network.",
)
@click.option(
    "--style", "-s",
    type=click.Choice(TUTORIAL_STYLES, case_sensitive=False),
    default=None,
    help=(
        "Generate an LLM-powered tutorial in one of 6 styles: "
        "steps (numbered lesson), cornell (two-column notes), "
        "flashcard (Q&A cards), timeline (chaptered), "
        "cookbook (recipe format), slides (presentation deck). "
        "Requires an Anthropic API key or a running Ollama server."
    ),
)
def get(
    videos: tuple[str, ...],
    fmt: str,
    lang: str | None,
    output: str | None,
    save: bool,
    db: str,
    open_browser: bool,
    tutor: bool,
    tutor_endpoint: str,
    tutor_ollama_url: str | None,
    tutor_discover: bool,
    style: str | None,
) -> None:
    """
    Fetch one or more YouTube video transcripts.

    VIDEO can be a full YouTube URL or an 11-character video ID.  Pass
    multiple URLs/IDs to combine transcripts (requires --style).
    By default, saves to the local DB and writes a markdown document to
    ~/Documents/yt-transcripts/{channel}/{title}.html.
    Use --no-save to skip DB persistence, or --format text/json for stdout.
    """
    from yt_transcript_extractor.extractor import parse_video_id, get_transcript

    # Validate: multiple videos only makes sense with --style.
    if len(videos) > 1 and not style:
        click.echo(
            "Error: Multiple videos require --style to combine them into a tutorial. "
            "Pass a single video for standard transcript extraction, or add "
            "--style <style> to generate a combined tutorial.",
            err=True,
        )
        sys.exit(1)

    # Parse the comma-separated language list into a proper list, if provided.
    languages: list[str] | None = None
    if lang:
        languages = [code.strip() for code in lang.split(",")]

    # --- Phase 1: Extract and save each video's transcript individually ---
    # This happens for all videos regardless of whether --style is used.
    # Each video gets saved to the DB and gets its own doc file (if applicable).
    # We also collect (title, transcript) pairs for multi-source tutorial
    # generation in Phase 2.
    video_ids: list[str] = []
    video_titles: list[str] = []
    first_result_text: str | None = None  # For single-video stdout output.

    for video in videos:
        try:
            result = extract(
                video,
                languages=languages,
                fmt=fmt,
                save=save,
                db_path=db if save else None,
            )
        except TranscriptError as exc:
            click.echo(f"Error: {exc.message}", err=True)
            sys.exit(1)

        vid = parse_video_id(video)
        video_ids.append(vid)

        # Look up the video title from the DB.  This is a read-only query, so
        # we attempt it regardless of --save: the video may already have been
        # saved by an earlier run, in which case we can recover the real title
        # even in --no-save mode (otherwise the tutorial would be titled the
        # generic "Transcript" fallback).
        vid_title = "Transcript"
        try:
            with TranscriptStore(db) as store:
                row = store.conn.execute(
                    "SELECT title FROM videos WHERE video_id = ?",
                    [vid],
                ).fetchone()
                if row:
                    vid_title = row[0]
        except TranscriptError:
            pass
        video_titles.append(vid_title)

        if save:
            click.echo(f"Transcript saved to {db} ({vid_title})", err=True)

        # For single-video mode, keep the result for output below.
        if len(videos) == 1:
            if isinstance(result, dict):
                first_result_text = json.dumps(result, indent=2, ensure_ascii=False)
            else:
                first_result_text = result

    # --- Phase 1b: Write single-video transcript output ---
    # Only applies when a single video is given (multi-video skips this
    # because the individual transcripts are just saved, not output).
    wrote_to_file = False
    if len(videos) == 1 and first_result_text is not None:
        text = first_result_text
        if output and not style:
            with open(output, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.write("\n")
            click.echo(f"Transcript written to {output}", err=True)
            wrote_to_file = True
        elif fmt == "doc" and save and not style:
            auto_path = _auto_output_path(video_ids[0], db)
            if auto_path:
                os.makedirs(os.path.dirname(auto_path), exist_ok=True)
                with open(auto_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                    fh.write("\n")
                click.echo(f"Transcript written to {auto_path}", err=True)
                wrote_to_file = True
                if open_browser:
                    subprocess.Popen(["chromium", auto_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            else:
                click.echo(text)

    # --- Phase 2: LLM-powered tutorial generation (--style) ---
    if style:
        # Resolve Ollama URL via network discovery if requested.
        if tutor_discover and not tutor_ollama_url:
            discovered_url = _discover_ollama()
            if discovered_url:
                click.echo(f"Discovered Ollama at {discovered_url}", err=True)
                tutor_ollama_url = discovered_url
            else:
                click.echo("No Ollama servers discovered.", err=True)

        # Fetch raw transcripts for each video (re-fetch because extract()
        # already consumed the FetchedTranscript for formatting).
        multi_sources: list[tuple[str, object]] = []
        for vid, vid_title in zip(video_ids, video_titles):
            raw_transcript = get_transcript(vid, languages=languages)
            multi_sources.append((vid_title, raw_transcript))

        # Choose the document title:
        # - Single video: use that video's title.
        # - Multiple videos: use a combined title.
        if len(videos) == 1:
            doc_title = video_titles[0]
        else:
            doc_title = " + ".join(video_titles)

        source_count = len(videos)
        click.echo(
            f"Generating '{style}' tutorial"
            f"{f' from {source_count} sources' if source_count > 1 else ''}"
            f" for '{doc_title}'...",
            err=True,
        )

        try:
            if len(multi_sources) == 1:
                # Single source — use the simpler code path.
                tutorial_html = format_tutorial(
                    multi_sources[0][1],
                    title=doc_title,
                    style=style,
                    endpoint=tutor_endpoint.lower(),
                    ollama_url=tutor_ollama_url,
                )
            else:
                # Multi-source — pass all transcripts for integrated generation.
                tutorial_html = format_tutorial(
                    title=doc_title,
                    style=style,
                    endpoint=tutor_endpoint.lower(),
                    ollama_url=tutor_ollama_url,
                    multi_sources=multi_sources,
                )
        except TranscriptError as exc:
            click.echo(f"Error: {exc.message}", err=True)
            sys.exit(1)

        # Determine output path.
        if output:
            out_path = output
        elif save:
            # For single video, use auto-path with style suffix.
            # For multi-video, build a path from the first video with
            # a "combined-{style}" suffix.
            base_auto = _auto_output_path(video_ids[0], db)
            if base_auto:
                root, ext = os.path.splitext(base_auto)
                if len(videos) > 1:
                    out_path = f"{root}-combined-{style}{ext}"
                else:
                    out_path = f"{root}-{style}{ext}"
            else:
                out_path = None
        else:
            out_path = None

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(tutorial_html)
                fh.write("\n")
            click.echo(f"Tutorial written to {out_path}", err=True)
            if open_browser:
                subprocess.Popen(
                    ["chromium", out_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        else:
            click.echo(tutorial_html)

    elif tutor:
        # Legacy --tutor flag — only supports single video.
        from yt_transcript_extractor.metadata import fetch_video_metadata
        from yt_transcript_extractor.extractor import format_text

        video_id = video_ids[0]
        metadata = fetch_video_metadata(video_id)
        if tutor_discover and not tutor_ollama_url:
            discovered_url = _discover_ollama()
            if discovered_url:
                click.echo(f"Discovered Ollama at {discovered_url}", err=True)
                tutor_ollama_url = discovered_url
            else:
                click.echo("No Ollama servers discovered.", err=True)

        raw_transcript = get_transcript(video_id, languages=languages)
        text_transcript = format_text(raw_transcript)

        tutor_script = os.path.expanduser("~/AI/rlm-based-tutor/video_tutor_gen.py")
        tutor_dir = os.path.expanduser("~/AI/rlm-based-tutor")

        click.echo(f"Generating tutorial for '{metadata.title}'...", err=True)

        cmd = [
            "uv", "run", "python", tutor_script,
            "--title", metadata.title,
            "--channel", metadata.channel_name,
            "--video-url", f"https://www.youtube.com/watch?v={video_id}",
            "--endpoint", tutor_endpoint.lower()
        ]

        if tutor_ollama_url:
            cmd.extend(["--ollama-url", tutor_ollama_url])

        try:
            subprocess.run(
                cmd,
                input=text_transcript,
                text=True,
                cwd=tutor_dir,
                check=True
            )
        except subprocess.CalledProcessError as exc:
            click.echo(f"Error: Tutorial generation failed: {exc}", err=True)
    elif not wrote_to_file:
        # Non-doc format or save is off — print to stdout as before.
        if first_result_text is not None:
            click.echo(first_result_text)


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
@click.argument("video_ids", metavar="VIDEO_ID...", nargs=-1, required=True)
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
@click.option(
    "--open/--no-open",
    "open_browser",
    default=False,
    help="Open the generated HTML file in Chromium.",
)
@click.option(
    "--style", "-s",
    type=click.Choice(TUTORIAL_STYLES, case_sensitive=False),
    default=None,
    help="Generate an LLM-powered tutorial (see 'get --help' for style descriptions).",
)
@click.option(
    "--tutor-endpoint",
    type=click.Choice(["auto", "anthropic", "ollama"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="LLM endpoint for tutorial generation (used with --style).",
)
@click.option(
    "--tutor-ollama-url",
    default=None,
    help="Custom Ollama server URL (used with --style).",
)
def saved(
    video_ids: tuple[str, ...],
    fmt: str,
    output: str | None,
    db: str,
    open_browser: bool,
    style: str | None,
    tutor_endpoint: str,
    tutor_ollama_url: str | None,
) -> None:
    """
    Retrieve one or more previously saved transcripts from the local database.

    VIDEO_ID is an 11-character YouTube video identifier.  Pass multiple IDs
    to combine transcripts (requires --style).
    This does NOT fetch from YouTube — it only reads from the local DB.
    """
    # Validate: multiple video IDs only makes sense with --style.
    if len(video_ids) > 1 and not style:
        click.echo(
            "Error: Multiple video IDs require --style to combine them into a tutorial.",
            err=True,
        )
        sys.exit(1)

    try:
        with TranscriptStore(db) as store:
            # Verify all videos exist in the DB.
            for vid in video_ids:
                if not store.has_video(vid):
                    click.echo(
                        f"Error: Video {vid} not found in database. "
                        f"Use 'yt-transcript get {vid}' to save it first.",
                        err=True,
                    )
                    sys.exit(1)

            if style:
                # --style: fetch raw segments for all videos, run through
                # the LLM tutorial pipeline (multi-source if > 1 video).
                multi_sources: list[tuple[str, list]] = []
                vid_titles: list[str] = []

                for vid in video_ids:
                    segments = store.get_transcript(vid)
                    vid_title = "Transcript"
                    row = store.conn.execute(
                        "SELECT title FROM videos WHERE video_id = ?",
                        [vid],
                    ).fetchone()
                    if row:
                        vid_title = row[0]
                    multi_sources.append((vid_title, segments))
                    vid_titles.append(vid_title)

                if len(video_ids) == 1:
                    doc_title = vid_titles[0]
                else:
                    doc_title = " + ".join(vid_titles)

                source_count = len(video_ids)
                click.echo(
                    f"Generating '{style}' tutorial"
                    f"{f' from {source_count} sources' if source_count > 1 else ''}"
                    f" for '{doc_title}'...",
                    err=True,
                )

                try:
                    if len(multi_sources) == 1:
                        result: str | dict = format_tutorial(
                            multi_sources[0][1],
                            title=doc_title,
                            style=style,
                            endpoint=tutor_endpoint.lower(),
                            ollama_url=tutor_ollama_url,
                        )
                    else:
                        result = format_tutorial(
                            title=doc_title,
                            style=style,
                            endpoint=tutor_endpoint.lower(),
                            ollama_url=tutor_ollama_url,
                            multi_sources=multi_sources,
                        )
                except TranscriptError as exc:
                    click.echo(f"Error: {exc.message}", err=True)
                    sys.exit(1)

                # Override fmt to "doc" so the output-path logic below
                # treats this as an HTML file.
                fmt = "doc"

            elif fmt == "json":
                # Single video only (validated above).
                segments = store.get_transcript(video_ids[0])
                result = {
                    "video_id": video_ids[0],
                    "segment_count": len(segments),
                    "segments": segments,
                }
            elif fmt == "doc":
                result = store.get_transcript_doc(video_ids[0])
            else:
                result = store.get_transcript_text(video_ids[0])
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
    elif fmt == "doc":
        auto_path = _auto_output_path(video_ids[0], db)

        if auto_path and style:
            root, ext = os.path.splitext(auto_path)
            if len(video_ids) > 1:
                auto_path = f"{root}-combined-{style}{ext}"
            else:
                auto_path = f"{root}-{style}{ext}"

        if auto_path:
            os.makedirs(os.path.dirname(auto_path), exist_ok=True)
            with open(auto_path, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.write("\n")
            click.echo(f"Transcript written to {auto_path}", err=True)

            if open_browser:
                subprocess.Popen(["chromium", auto_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        else:
            click.echo(text)
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


# ---------------------------------------------------------------------------
# Subcommand: yt-search — search YouTube and optionally generate tutorials
# ---------------------------------------------------------------------------

@main.command("yt-search")
@click.argument("query")
@click.option(
    "-n", "--count",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of videos to return.",
)
@click.option(
    "-p", "--period",
    type=click.Choice(list(PERIOD_MAP.keys()), case_sensitive=False),
    default=None,
    help="Only include videos uploaded within this time period.",
)
@click.option(
    "--style", "-s",
    type=click.Choice(TUTORIAL_STYLES, case_sensitive=False),
    default=None,
    help=(
        "Automatically extract transcripts from search results and generate "
        "a combined tutorial.  See 'get --help' for style descriptions."
    ),
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Save extracted transcripts to the local database (used with --style).",
)
@click.option(
    "--db",
    default=_DEFAULT_DB,
    show_default=True,
    help="Path to the DuckDB database file (used with --style --save).",
)
@click.option(
    "--lang", "-l",
    default=None,
    help="Comma-separated language codes for transcript extraction (used with --style).",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Write tutorial output to this file (used with --style).",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=False,
    help="Open the generated HTML file in Chromium (used with --style).",
)
@click.option(
    "--tutor-endpoint",
    type=click.Choice(["auto", "anthropic", "ollama"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="LLM endpoint for tutorial generation (used with --style).",
)
@click.option(
    "--tutor-ollama-url",
    default=None,
    help="Custom Ollama server URL (used with --style).",
)
@click.option(
    "--tutor-discover/--no-tutor-discover",
    default=False,
    help="Automatically discover Ollama server on local network.",
)
def yt_search(
    query: str,
    count: int,
    period: str | None,
    style: str | None,
    save: bool,
    db: str,
    lang: str | None,
    output: str | None,
    open_browser: bool,
    tutor_endpoint: str,
    tutor_ollama_url: str | None,
    tutor_discover: bool,
) -> None:
    """
    Search YouTube for videos and optionally generate a tutorial.

    QUERY is the search term (e.g. "generative chemistry").

    Without --style, lists matching videos with metadata.
    With --style, extracts transcripts from all results and combines them
    into a single integrated tutorial.

    \b
    Examples:
      yt-transcript yt-search "generative chemistry"
      yt-transcript yt-search "rust programming" -p "this month" -n 3
      yt-transcript yt-search "docker tutorial" -p "past 3 months" -s steps
      yt-transcript yt-search "machine learning" -n 5 -s slides
    """
    # Step 1: Search YouTube.
    period_label = f" ({period})" if period else ""
    click.echo(f"Searching YouTube for '{query}'{period_label}...", err=True)

    try:
        results = search_youtube(query, max_results=count, period=period)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except TranscriptError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)

    if not results:
        click.echo("No videos found matching your search.")
        return

    # Step 2: Display results.
    click.echo(f"\nFound {len(results)} video{'s' if len(results) != 1 else ''}:\n", err=True)
    for i, r in enumerate(results, 1):
        date_str = str(r.upload_date) if r.upload_date else "unknown date"
        duration = format_duration(r.duration_secs)
        views = format_view_count(r.view_count)
        views_str = f"  {views}" if views else ""

        click.echo(f"  {i}. {r.title}", err=True)
        click.echo(f"     {r.channel_name}  |  {date_str}  |  {duration}{views_str}", err=True)
        click.echo(f"     ID: {r.video_id}", err=True)
        click.echo(err=True)

    # If no --style, we're done — just listing results.
    if not style:
        return

    # Step 3: Extract transcripts for all search results.
    click.echo(f"Extracting transcripts from {len(results)} videos...", err=True)

    languages: list[str] | None = None
    if lang:
        languages = [code.strip() for code in lang.split(",")]

    # Extract and save each video.  Collect (title, transcript) pairs
    # for multi-source tutorial generation.
    from yt_transcript_extractor.extractor import get_transcript

    multi_sources: list[tuple[str, object]] = []
    video_ids: list[str] = []
    video_titles: list[str] = []
    failed: list[str] = []

    for r in results:
        try:
            # Save to DB via extract() for persistence.
            extract(
                r.video_id,
                languages=languages,
                fmt="text",
                save=save,
                db_path=db if save else None,
            )
            # Re-fetch raw transcript for the tutorial pipeline.
            raw_transcript = get_transcript(r.video_id, languages=languages)
            multi_sources.append((r.title, raw_transcript))
            video_ids.append(r.video_id)
            video_titles.append(r.title)
            click.echo(f"  Extracted: {r.title}", err=True)
        except TranscriptError as exc:
            # Some videos may not have transcripts — skip them gracefully.
            click.echo(f"  Skipped: {r.title} ({exc.message})", err=True)
            failed.append(r.title)

    if not multi_sources:
        click.echo("Error: No transcripts could be extracted from search results.", err=True)
        sys.exit(1)

    if failed:
        click.echo(
            f"\n{len(failed)} video(s) skipped (no transcript available).",
            err=True,
        )

    # Step 4: Resolve Ollama URL if needed.
    if tutor_discover and not tutor_ollama_url:
        discovered_url = _discover_ollama()
        if discovered_url:
            click.echo(f"Discovered Ollama at {discovered_url}", err=True)
            tutor_ollama_url = discovered_url
        else:
            click.echo("No Ollama servers discovered.", err=True)

    # Step 5: Generate the combined tutorial.
    doc_title = query.title()  # Use the search query as the document title.
    source_count = len(multi_sources)
    click.echo(
        f"\nGenerating '{style}' tutorial from {source_count} source"
        f"{'s' if source_count != 1 else ''}...",
        err=True,
    )

    try:
        if len(multi_sources) == 1:
            tutorial_html = format_tutorial(
                multi_sources[0][1],
                title=doc_title,
                style=style,
                endpoint=tutor_endpoint.lower(),
                ollama_url=tutor_ollama_url,
            )
        else:
            tutorial_html = format_tutorial(
                title=doc_title,
                style=style,
                endpoint=tutor_endpoint.lower(),
                ollama_url=tutor_ollama_url,
                multi_sources=multi_sources,
            )
    except TranscriptError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)

    # Step 6: Write output.
    if output:
        out_path = output
    else:
        # Auto-generate a path based on the search query.
        base = os.path.expanduser(_AUTO_OUTPUT_BASE)
        sanitized_query = _sanitize_filename(query)
        out_dir = os.path.join(base, "_search")
        out_path = os.path.join(out_dir, f"{sanitized_query}-{style}.html")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(tutorial_html)
        fh.write("\n")
    click.echo(f"Tutorial written to {out_path}", err=True)

    if open_browser:
        subprocess.Popen(
            ["chromium", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
