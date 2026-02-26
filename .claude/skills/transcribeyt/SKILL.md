---
name: transcribeyt
description: Fetch a YouTube video transcript and place it into the conversation context. Triggers on "YouTube transcript", "video transcript", "fetch transcript".
user_invocable: true
tools:
  - Bash
---

# YouTube Transcript to Context

When invoked with `/transcribeyt <URL_OR_ID>`, fetch the transcript as plain text and inject it into the conversation context.

## Instructions

1. The user provides a YouTube URL or 11-character video ID as the argument.
2. Run the following command via Bash:

   ```
   uv run yt-transcript get "<URL_OR_ID>" -f text --no-save
   ```

   - `-f text` outputs plain text (no timestamps, no HTML).
   - `--no-save` skips DuckDB persistence — this is ephemeral context only.

3. The transcript text printed to stdout becomes part of the conversation context automatically.
4. Do **not** write any files, save to the database, or create documents. The sole purpose is to get the transcript text into context so the user can ask questions about it, summarize it, etc.
5. After the transcript loads, briefly confirm it's in context (e.g. "Transcript loaded — N lines from **Video Title** by **Channel**." if that info is visible, otherwise just "Transcript loaded.") and ask how the user would like to work with it.
