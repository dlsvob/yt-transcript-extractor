"""
api.py — FastAPI REST API for yt-transcript-extractor.

Provides endpoints for fetching transcripts from YouTube, as well as
querying locally-saved transcripts organized by channel.

Endpoints:
    GET /transcript/{video_id}          — Fetch a transcript (text or JSON), optionally saving it.
    GET /channels                       — List all channels with saved transcripts.
    GET /channels/{channel_id}/videos   — List saved videos for a specific channel.
    GET /saved/{video_id}               — Retrieve a previously saved transcript.
    GET /search                         — Search across all saved transcript segments.
    GET /health                         — Simple health-check for load balancers / monitoring.

Run with:
    uv run uvicorn yt_transcript_extractor.api:app

The global exception handler catches any TranscriptError and converts it to
the appropriate HTTP response using the status code stored on the exception.
"""

from __future__ import annotations

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from yt_transcript_extractor.errors import TranscriptError
from yt_transcript_extractor.extractor import extract
from yt_transcript_extractor.storage import TranscriptStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default DuckDB database path, shared across all endpoints that access storage.
_DEFAULT_DB = "transcripts.duckdb"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="YouTube Transcript Extractor API",
    description="Extract YouTube video transcripts as plain text or structured JSON. "
                "Supports saving transcripts locally for offline retrieval and search.",
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(TranscriptError)
async def transcript_error_handler(request: Request, exc: TranscriptError) -> JSONResponse:
    """
    Translate any TranscriptError (or subclass) into an HTTP error response.

    The http_status on the exception drives the response code, so individual
    endpoint code never needs to think about HTTP semantics — it just raises
    the right library exception and this handler takes care of the rest.
    This covers all error subclasses including MetadataFetchError (502) and
    StorageError (500).
    """
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.message},
    )


# ---------------------------------------------------------------------------
# Endpoints — transcript fetching
# ---------------------------------------------------------------------------

# response_model=None is required because we return different Response subclasses
# (PlainTextResponse or JSONResponse) depending on the format param, and FastAPI
# can't build a Pydantic model from a Union of Response types.
@app.get("/transcript/{video_id}", response_model=None)
async def get_transcript(
    video_id: str,
    format: str = Query(
        default="text",
        description="Output format: 'text' for plain transcript, 'json' for structured data with timestamps, 'doc' for readable markdown document.",
        pattern="^(text|json|doc)$",
    ),
    lang: str = Query(
        default="",
        description="Comma-separated language codes in priority order (e.g. 'de,en'). Empty defaults to English.",
    ),
    save: bool = Query(
        default=False,
        description="If true, persist the transcript to a local DuckDB database.",
    ),
    db: str = Query(
        default=_DEFAULT_DB,
        description="Path to the DuckDB database file (only used when save=true).",
    ),
) -> PlainTextResponse | JSONResponse:
    """
    Fetch the transcript for a single YouTube video.

    **video_id** is the 11-character YouTube video identifier
    (e.g. `dQw4w9WgXcQ`).

    The response format depends on the `format` query parameter:
    - `text` (default): plain text, one line per caption segment.
    - `json`: a JSON object with `video_id`, `segment_count`, and a
      `segments` array where each entry has `text`, `start`, `duration`.

    When `save=true`, the transcript and video metadata are also stored in
    a local DuckDB database for offline retrieval via the /saved endpoint.
    """
    # Build language list from the comma-separated query param.
    languages: list[str] | None = None
    if lang:
        languages = [code.strip() for code in lang.split(",")]

    # extract() may raise TranscriptError subclasses — the global handler
    # will convert those into the correct HTTP error response.
    result = extract(
        video_id,
        languages=languages,
        fmt=format,
        save=save,
        db_path=db if save else None,
    )

    # Return plain text or JSON depending on the requested format.
    if isinstance(result, dict):
        return JSONResponse(content=result)
    return PlainTextResponse(content=result)


# ---------------------------------------------------------------------------
# Endpoints — saved transcript queries
# ---------------------------------------------------------------------------

@app.get("/channels")
async def list_channels(
    db: str = Query(
        default=_DEFAULT_DB,
        description="Path to the DuckDB database file.",
    ),
) -> JSONResponse:
    """
    List all channels that have saved transcripts.

    Returns an array of channel objects, each with the channel's name, ID,
    URL, and the number of saved videos.
    """
    with TranscriptStore(db) as store:
        channel_list = store.list_channels()

    return JSONResponse(content={
        "channels": [
            {
                "channel_id": ch.channel_id,
                "channel_name": ch.channel_name,
                "channel_url": ch.channel_url,
                "video_count": ch.video_count,
            }
            for ch in channel_list
        ],
    })


@app.get("/channels/{channel_id}/videos")
async def list_videos(
    channel_id: str,
    db: str = Query(
        default=_DEFAULT_DB,
        description="Path to the DuckDB database file.",
    ),
) -> JSONResponse:
    """
    List all saved videos for a specific channel.

    **channel_id** is YouTube's internal channel identifier (e.g. `UC38IQsAvIsxxjztdMZQtwHA`).
    Use the `/channels` endpoint to discover channel IDs.
    """
    with TranscriptStore(db) as store:
        video_list = store.list_videos(channel_id)

    return JSONResponse(content={
        "channel_id": channel_id,
        "videos": [
            {
                "video_id": v.video_id,
                "title": v.title,
                "upload_date": str(v.upload_date) if v.upload_date else None,
                "duration_secs": v.duration_secs,
                "language": v.language,
                "language_code": v.language_code,
                "is_generated": v.is_generated,
            }
            for v in video_list
        ],
    })


@app.get("/saved/{video_id}", response_model=None)
async def get_saved_transcript(
    video_id: str,
    format: str = Query(
        default="text",
        description="Output format: 'text' for plain transcript, 'json' for structured data with timestamps, 'doc' for readable markdown document.",
        pattern="^(text|json|doc)$",
    ),
    db: str = Query(
        default=_DEFAULT_DB,
        description="Path to the DuckDB database file.",
    ),
) -> PlainTextResponse | JSONResponse:
    """
    Retrieve a previously saved transcript from the local database.

    This does NOT fetch from YouTube — it only reads from the local DuckDB
    store.  Use the `GET /transcript/{video_id}?save=true` endpoint to save
    a transcript first.
    """
    with TranscriptStore(db) as store:
        if not store.has_video(video_id):
            return JSONResponse(
                status_code=404,
                content={"error": f"Video {video_id} not found in database."},
            )

        if format == "json":
            segments = store.get_transcript(video_id)
            return JSONResponse(content={
                "video_id": video_id,
                "segment_count": len(segments),
                "segments": segments,
            })
        elif format == "doc":
            # Readable markdown document with timestamped paragraphs.
            text = store.get_transcript_doc(video_id)
            return PlainTextResponse(content=text)
        else:
            text = store.get_transcript_text(video_id)
            return PlainTextResponse(content=text)


@app.get("/search")
async def search_transcripts(
    q: str = Query(
        description="Case-insensitive search term to match against transcript text.",
    ),
    db: str = Query(
        default=_DEFAULT_DB,
        description="Path to the DuckDB database file.",
    ),
) -> JSONResponse:
    """
    Search across all saved transcripts for matching segments.

    Returns an array of matching segments, each with the video context
    (title, channel name) and timestamp information.
    """
    with TranscriptStore(db) as store:
        results = store.search_transcripts(q)

    return JSONResponse(content={
        "query": q,
        "result_count": len(results),
        "results": results,
    })


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """
    Minimal health-check endpoint.

    Returns HTTP 200 with {"status": "ok"}.  Useful for container
    orchestrators, load balancers, and uptime monitoring.
    """
    return {"status": "ok"}
