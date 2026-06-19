"""
tutorials.py — LLM-powered tutorial generation from video transcripts.

This module converts raw transcript text into structured data suitable for
rendering as styled HTML tutorials.  It sits between the raw transcript
(plain text) and the HTML templates (in templates.py):

    raw transcript text  →  tutorials.py (LLM structuring)  →  templates.py (HTML rendering)

Six tutorial styles are supported, each producing a different data structure
that captures the pedagogical layout:

    steps     — Numbered step-by-step lesson with key concepts per step.
    cornell   — Cornell notes: cue column + notes column + summary.
    flashcard — Question-and-answer cards for retention practice.
    timeline  — Chaptered timeline with auto-detected topic boundaries.
    cookbook   — Recipe/cookbook format: prerequisites, instructions, notes.
    slides    — Slide-deck format: one major topic per "slide".

The module supports two LLM backends:
    - Anthropic (Claude API) — used when ANTHROPIC_API_KEY is set or
      explicitly selected via --tutor-endpoint anthropic.
    - Ollama (local models) — used when an Ollama server URL is provided
      or discovered on the local network.

The "auto" endpoint (default) tries Anthropic first, then falls back to Ollama.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from yt_transcript_extractor.errors import LLMError


# ---------------------------------------------------------------------------
# Constants — list of valid style names, used for CLI validation too
# ---------------------------------------------------------------------------

# All supported tutorial styles.  This list is the single source of truth —
# the CLI reads it for --style choices, and generate_tutorial() dispatches on it.
TUTORIAL_STYLES = ["steps", "cornell", "flashcard", "timeline", "cookbook", "slides"]

# Default Anthropic model to use for tutorial generation.
# Claude Sonnet 4.6 is fast and capable enough for structured extraction.
# (The previous pinned ID "claude-sonnet-4-20250514" was retired and now
# returns a 404 not_found_error from the API.)
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Default Ollama model — a capable open model for structured output.
_DEFAULT_OLLAMA_MODEL = "llama3.1"


# ---------------------------------------------------------------------------
# Data structures — one dataclass per tutorial style
# ---------------------------------------------------------------------------
# Each dataclass represents the structured output the LLM produces.
# The LLM returns JSON matching these shapes; we parse it into dataclasses
# so the template renderers have typed, predictable inputs.

@dataclass
class Step:
    """A single step in a step-by-step tutorial.

    Attributes:
        number:       The step's position in the sequence (1-based).
        title:        Short heading describing what this step covers.
        key_concepts: List of 1-3 key takeaways from this step.
        content:      The full explanatory text for this step.
    """
    number: int
    title: str
    key_concepts: list[str]
    content: str


@dataclass
class StepsStructure:
    """Complete structure for the 'steps' tutorial style.

    A numbered sequence of lesson steps, each with a title, key concepts,
    and detailed content extracted from the transcript.

    Attributes:
        steps:   Ordered list of Step objects.
        summary: A 2-3 sentence overview of the entire tutorial.
    """
    steps: list[Step]
    summary: str


@dataclass
class CornellNote:
    """A single row in a Cornell notes layout.

    The Cornell method uses two columns: a narrow 'cue' column on the left
    (keywords, questions, prompts) and a wider 'notes' column on the right
    (detailed notes, explanations, transcript content).

    Attributes:
        cue:   A keyword, question, or prompt for the left column.
        notes: Detailed notes / transcript content for the right column.
    """
    cue: str
    notes: str


@dataclass
class CornellStructure:
    """Complete structure for the 'cornell' tutorial style.

    Two-column Cornell notes with a bottom summary section.

    Attributes:
        notes:   List of CornellNote rows (cue + notes pairs).
        summary: Bottom-of-page summary capturing the key takeaways.
    """
    notes: list[CornellNote]
    summary: str


@dataclass
class Flashcard:
    """A single flashcard with a question and answer.

    Rendered as a clickable card that reveals the answer on click/tap.

    Attributes:
        question: The question or prompt shown on the card front.
        answer:   The answer revealed when the card is flipped/clicked.
    """
    question: str
    answer: str


@dataclass
class FlashcardStructure:
    """Complete structure for the 'flashcard' tutorial style.

    A collection of Q&A flashcards extracted from the transcript content.

    Attributes:
        cards: List of Flashcard objects (question + answer pairs).
        topic: A short label describing the overall topic of the cards.
    """
    cards: list[Flashcard]
    topic: str


@dataclass
class Chapter:
    """A single chapter in a timeline layout.

    Represents a distinct topic segment within the video, with its own
    timestamp, title, summary, and detailed content.

    Attributes:
        timestamp: The MM:SS timestamp where this chapter starts in the video.
        title:     Short heading for this chapter/topic.
        summary:   A 1-2 sentence overview of what's covered.
        content:   The full detailed content for this chapter.
    """
    timestamp: str
    title: str
    summary: str
    content: str


@dataclass
class TimelineStructure:
    """Complete structure for the 'timeline' tutorial style.

    A sequence of chapters with timestamps, suitable for rendering as a
    vertical or horizontal timeline with keyboard navigation.

    Attributes:
        chapters: Ordered list of Chapter objects.
        overview: A brief overview of the entire video's content arc.
    """
    chapters: list[Chapter]
    overview: str


@dataclass
class CookbookStructure:
    """Complete structure for the 'cookbook' tutorial style.

    Modeled after a recipe card: what you need before starting, the actual
    step-by-step instructions, and additional tips/warnings.

    Attributes:
        prerequisites: Things needed before starting (tools, knowledge, setup).
        instructions:  Numbered list of action steps to follow.
        notes:         Tips, warnings, gotchas, and additional context.
        difficulty:    Estimated difficulty level (beginner/intermediate/advanced).
    """
    prerequisites: list[str]
    instructions: list[str]
    notes: list[str]
    difficulty: str


@dataclass
class Slide:
    """A single slide in a presentation-style layout.

    Each slide covers one major topic and contains a heading plus
    a few bullet points — like a real presentation slide.

    Attributes:
        heading: The slide's title/heading.
        bullets: List of 3-5 key points shown as bullet items.
    """
    heading: str
    bullets: list[str]


@dataclass
class SlidesStructure:
    """Complete structure for the 'slides' tutorial style.

    A sequence of slides, each covering one major topic from the video.
    Rendered with CSS scroll-snap for slide-like navigation.

    Attributes:
        slides:      Ordered list of Slide objects.
        title_slide: A short subtitle for the title/cover slide.
    """
    slides: list[Slide]
    title_slide: str


# ---------------------------------------------------------------------------
# LLM prompt builders — one per tutorial style
# ---------------------------------------------------------------------------
# Each function returns a (system_prompt, user_prompt) tuple.
# The system prompt defines the task and expected JSON schema.
# The user prompt contains one or more video sources with their transcripts.
#
# All prompt builders accept a list of (title, transcript_text) pairs,
# supporting both single-video and multi-video tutorial generation.
# When multiple sources are provided, the system prompt includes integration
# instructions: merge overlapping topics into unified sections, keep
# orthogonal (unrelated) content in separate sections.

# Type alias for a list of (title, transcript_text) source pairs.
# Each pair represents one video's contribution to the tutorial.
SourceList = list[tuple[str, str]]


def _format_sources(sources: SourceList) -> str:
    """Format one or more transcript sources into a labeled user prompt body.

    For a single source, just includes the title and transcript text.
    For multiple sources, wraps each in a clearly labeled section so the
    LLM can distinguish which content came from which video.

    Args:
        sources: List of (title, transcript_text) pairs.

    Returns:
        A formatted string ready to embed in the user prompt.
    """
    if len(sources) == 1:
        title, text = sources[0]
        return f'Video: "{title}"\n\n{text}'

    # Multiple sources — label each one clearly.
    parts = []
    for i, (title, text) in enumerate(sources, 1):
        parts.append(f'--- Source {i}: "{title}" ---\n{text}')
    return "\n\n".join(parts)


def _multi_source_instruction(source_count: int) -> str:
    """Return integration instructions for multi-source prompts.

    Only returns non-empty text when there are multiple sources.
    The instruction tells the LLM to integrate overlapping topics into
    unified sections while keeping unrelated content separate — this
    naturally handles the orthogonality spectrum without extra code.

    Args:
        source_count: Number of transcript sources being combined.

    Returns:
        An instruction string to prepend to the system prompt, or empty
        string for single-source prompts.
    """
    if source_count <= 1:
        return ""
    return (
        "You are given transcripts from MULTIPLE video sources. "
        "Integrate content that covers overlapping or related topics into "
        "unified, cohesive sections — do not duplicate similar information. "
        "Content that covers distinct, unrelated topics should remain in "
        "separate sections, clearly delineated. "
        "The more the sources overlap in topic, the more tightly integrated "
        "the output should be; the more orthogonal (unrelated) they are, the "
        "more the output should read like distinct, labeled sections. "
    )


def _build_steps_prompt(sources: SourceList) -> tuple[str, str]:
    """Build the LLM prompts for the 'steps' tutorial style.

    Instructs the model to break the transcript(s) into logical numbered
    steps, each with a title, key concepts, and explanatory content.
    When multiple sources are provided, steps should integrate overlapping
    content and keep orthogonal topics in separate steps.

    Args:
        sources: List of (title, transcript_text) pairs.

    Returns:
        A (system_prompt, user_prompt) tuple ready for the LLM call.
    """
    system = (
        _multi_source_instruction(len(sources)) +
        "You extract structured lesson steps from video transcripts. "
        "Break the content into logical, numbered steps that a learner would follow. "
        "Each step should have a clear title, 1-3 key concepts, and detailed content. "
        "Return ONLY valid JSON matching this schema:\n"
        '{"steps": [{"number": 1, "title": "...", "key_concepts": ["..."], "content": "..."}], '
        '"summary": "2-3 sentence overview"}'
    )
    user = f"Extract lesson steps from this content:\n\n{_format_sources(sources)}"
    return system, user


def _build_cornell_prompt(sources: SourceList) -> tuple[str, str]:
    """Build the LLM prompts for the 'cornell' tutorial style.

    Instructs the model to produce Cornell-method notes: cue words/questions
    in the left column, detailed notes in the right column, plus a summary.

    Args:
        sources: List of (title, transcript_text) pairs.

    Returns:
        A (system_prompt, user_prompt) tuple.
    """
    system = (
        _multi_source_instruction(len(sources)) +
        "You create Cornell-method study notes from video transcripts. "
        "For each major topic, produce a 'cue' (a keyword, question, or prompt for the left column) "
        "and 'notes' (detailed explanation for the right column). "
        "End with a summary capturing the key takeaways. "
        "Return ONLY valid JSON matching this schema:\n"
        '{"notes": [{"cue": "...", "notes": "..."}], "summary": "..."}'
    )
    user = f"Create Cornell notes from this content:\n\n{_format_sources(sources)}"
    return system, user


def _build_flashcard_prompt(sources: SourceList) -> tuple[str, str]:
    """Build the LLM prompts for the 'flashcard' tutorial style.

    Instructs the model to extract key concepts and generate question-answer
    pairs suitable for study flashcards.

    Args:
        sources: List of (title, transcript_text) pairs.

    Returns:
        A (system_prompt, user_prompt) tuple.
    """
    system = (
        _multi_source_instruction(len(sources)) +
        "You extract key concepts from video transcripts and create study flashcards. "
        "Each flashcard has a clear question on the front and a concise answer on the back. "
        "Create 10-20 flashcards covering the most important concepts. "
        "Return ONLY valid JSON matching this schema:\n"
        '{"cards": [{"question": "...", "answer": "..."}], "topic": "short topic label"}'
    )
    user = f"Create flashcards from this content:\n\n{_format_sources(sources)}"
    return system, user


def _build_timeline_prompt(sources: SourceList) -> tuple[str, str]:
    """Build the LLM prompts for the 'timeline' tutorial style.

    Instructs the model to identify distinct topic chapters.  For multi-source
    input, chapters are organized by topic rather than by source video — the
    LLM interleaves content from different sources where topics overlap.

    Args:
        sources: List of (title, transcript_text) pairs.

    Returns:
        A (system_prompt, user_prompt) tuple.
    """
    system = (
        _multi_source_instruction(len(sources)) +
        "You identify distinct topic chapters in video transcripts. "
        "Break the content into logical chapters based on topic shifts. "
        "Each chapter needs a timestamp (MM:SS — estimate based on position in the transcript), "
        "a short title, a 1-2 sentence summary, and the detailed content. "
        "Return ONLY valid JSON matching this schema:\n"
        '{"chapters": [{"timestamp": "00:00", "title": "...", "summary": "...", "content": "..."}], '
        '"overview": "brief overview of the content"}'
    )
    user = f"Identify chapters in this content:\n\n{_format_sources(sources)}"
    return system, user


def _build_cookbook_prompt(sources: SourceList) -> tuple[str, str]:
    """Build the LLM prompts for the 'cookbook' tutorial style.

    Instructs the model to extract a recipe-card structure: prerequisites,
    step-by-step instructions, and tips/notes.  For multi-source input,
    prerequisites and instructions are unified across sources.

    Args:
        sources: List of (title, transcript_text) pairs.

    Returns:
        A (system_prompt, user_prompt) tuple.
    """
    system = (
        _multi_source_instruction(len(sources)) +
        "You extract practical how-to instructions from video transcripts, "
        "formatted like a recipe card. Identify: "
        "(1) prerequisites — tools, knowledge, or setup needed before starting; "
        "(2) instructions — numbered action steps the viewer should follow; "
        "(3) notes — tips, warnings, and gotchas mentioned in the video; "
        "(4) difficulty — beginner, intermediate, or advanced. "
        "Return ONLY valid JSON matching this schema:\n"
        '{"prerequisites": ["..."], "instructions": ["..."], "notes": ["..."], '
        '"difficulty": "beginner|intermediate|advanced"}'
    )
    user = f"Extract a how-to recipe from this content:\n\n{_format_sources(sources)}"
    return system, user


def _build_slides_prompt(sources: SourceList) -> tuple[str, str]:
    """Build the LLM prompts for the 'slides' tutorial style.

    Instructs the model to create a slide-deck breakdown: each "slide"
    covers one major topic with a heading and 3-5 bullet points.

    Args:
        sources: List of (title, transcript_text) pairs.

    Returns:
        A (system_prompt, user_prompt) tuple.
    """
    system = (
        _multi_source_instruction(len(sources)) +
        "You create presentation slide decks from video transcripts. "
        "Each slide covers one major topic with a clear heading and 3-5 bullet points. "
        "Keep bullets concise — they should work as presentation talking points. "
        "Create 5-15 slides depending on content length. "
        "Return ONLY valid JSON matching this schema:\n"
        '{"slides": [{"heading": "...", "bullets": ["...", "..."]}], '
        '"title_slide": "short subtitle for the cover slide"}'
    )
    user = f"Create presentation slides from this content:\n\n{_format_sources(sources)}"
    return system, user


# Map style names to their prompt builder functions.
_PROMPT_BUILDERS: dict[str, callable] = {
    "steps": _build_steps_prompt,
    "cornell": _build_cornell_prompt,
    "flashcard": _build_flashcard_prompt,
    "timeline": _build_timeline_prompt,
    "cookbook": _build_cookbook_prompt,
    "slides": _build_slides_prompt,
}


# ---------------------------------------------------------------------------
# LLM client — supports Anthropic and Ollama backends
# ---------------------------------------------------------------------------

def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Send a structured extraction request to the Anthropic Claude API.

    Uses the anthropic Python SDK.  Requires the ANTHROPIC_API_KEY environment
    variable to be set.  Requests JSON output by instructing the model in the
    system prompt (Claude reliably follows JSON-only instructions).

    Args:
        system_prompt: The system message defining the extraction task and schema.
        user_prompt:   The user message containing the video title and transcript.

    Returns:
        The raw text content of the model's response (should be JSON).

    Raises:
        LLMError: If the API key is missing, the request fails, or the
                  response is empty.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it or use --tutor-endpoint ollama for local models."
        )

    try:
        import anthropic
    except ImportError:
        raise LLMError(
            "The 'anthropic' package is not installed. "
            "Run: uv add anthropic"
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=_DEFAULT_ANTHROPIC_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Extract the text content from the response.
        text = response.content[0].text
        if not text:
            raise LLMError("Anthropic API returned an empty response.")
        return text

    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Anthropic API call failed: {exc}") from exc


def _call_ollama(system_prompt: str, user_prompt: str, ollama_url: str) -> str:
    """Send a structured extraction request to a local Ollama server.

    Uses httpx to make a direct REST call to the Ollama /api/generate endpoint.
    Requests JSON format output so the model returns parseable structured data.

    Args:
        system_prompt: The system message defining the extraction task and schema.
        user_prompt:   The user message containing the video title and transcript.
        ollama_url:    Base URL of the Ollama server (e.g. "http://localhost:11434").

    Returns:
        The raw text content of the model's response (should be JSON).

    Raises:
        LLMError: If the server is unreachable, returns an error, or the
                  response can't be parsed.
    """
    try:
        import httpx
    except ImportError:
        raise LLMError(
            "The 'httpx' package is not installed. "
            "Run: uv add httpx"
        )

    # Combine system and user prompts into a single prompt for Ollama's
    # /api/generate endpoint.  The "format: json" option tells Ollama
    # to constrain output to valid JSON.
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": _DEFAULT_OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "format": "json",
        "stream": False,
    }

    try:
        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if not text:
            raise LLMError("Ollama returned an empty response.")
        return text

    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Ollama API call failed: {exc}") from exc


def call_llm(
    system_prompt: str,
    user_prompt: str,
    endpoint: str = "auto",
    ollama_url: str | None = None,
) -> str:
    """Route an LLM request to the appropriate backend.

    This is the main LLM entry point used by generate_tutorial().  It handles
    endpoint selection logic:
      - "anthropic": use Claude API directly.
      - "ollama": use a local Ollama server.
      - "auto" (default): try Anthropic first (if API key is set), fall back to Ollama.

    Args:
        system_prompt: The system message defining the task.
        user_prompt:   The user message with the content to process.
        endpoint:      Which backend to use: "auto", "anthropic", or "ollama".
        ollama_url:    Base URL for the Ollama server (required if endpoint is "ollama",
                       optional for "auto" — defaults to http://localhost:11434).

    Returns:
        The raw text response from the LLM (expected to be JSON).

    Raises:
        LLMError: If no backend is available or the call fails.
    """
    if endpoint == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)

    if endpoint == "ollama":
        url = ollama_url or "http://localhost:11434"
        return _call_ollama(system_prompt, user_prompt, url)

    # "auto" mode: try Anthropic if API key exists, otherwise Ollama.
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _call_anthropic(system_prompt, user_prompt)

    url = ollama_url or "http://localhost:11434"
    return _call_ollama(system_prompt, user_prompt, url)


# ---------------------------------------------------------------------------
# JSON parsing — convert raw LLM output into typed dataclasses
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """Parse a JSON string from LLM output, handling common formatting issues.

    LLMs sometimes wrap JSON in markdown code fences (```json ... ```).
    This function strips those wrappers before parsing.

    Args:
        raw: The raw string from the LLM response.

    Returns:
        The parsed JSON as a Python dict.

    Raises:
        LLMError: If the string can't be parsed as valid JSON.
    """
    # Strip markdown code fences if the LLM wrapped its output.
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"LLM returned invalid JSON: {exc}. "
            f"First 200 chars of response: {raw[:200]}"
        ) from exc


def _parse_steps(data: dict) -> StepsStructure:
    """Parse a JSON dict into a StepsStructure dataclass.

    Validates that the expected keys exist and builds typed Step objects.

    Args:
        data: Parsed JSON dict from the LLM response.

    Returns:
        A StepsStructure with all steps and the summary.

    Raises:
        LLMError: If required keys are missing or malformed.
    """
    try:
        steps = [
            Step(
                number=s.get("number", i + 1),
                title=s["title"],
                key_concepts=s.get("key_concepts", []),
                content=s["content"],
            )
            for i, s in enumerate(data["steps"])
        ]
        return StepsStructure(steps=steps, summary=data.get("summary", ""))
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Failed to parse 'steps' structure: {exc}") from exc


def _parse_cornell(data: dict) -> CornellStructure:
    """Parse a JSON dict into a CornellStructure dataclass.

    Args:
        data: Parsed JSON dict from the LLM response.

    Returns:
        A CornellStructure with note rows and summary.

    Raises:
        LLMError: If required keys are missing.
    """
    try:
        notes = [
            CornellNote(cue=n["cue"], notes=n["notes"])
            for n in data["notes"]
        ]
        return CornellStructure(notes=notes, summary=data.get("summary", ""))
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Failed to parse 'cornell' structure: {exc}") from exc


def _parse_flashcard(data: dict) -> FlashcardStructure:
    """Parse a JSON dict into a FlashcardStructure dataclass.

    Args:
        data: Parsed JSON dict from the LLM response.

    Returns:
        A FlashcardStructure with cards and topic label.

    Raises:
        LLMError: If required keys are missing.
    """
    try:
        cards = [
            Flashcard(question=c["question"], answer=c["answer"])
            for c in data["cards"]
        ]
        return FlashcardStructure(cards=cards, topic=data.get("topic", ""))
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Failed to parse 'flashcard' structure: {exc}") from exc


def _parse_timeline(data: dict) -> TimelineStructure:
    """Parse a JSON dict into a TimelineStructure dataclass.

    Args:
        data: Parsed JSON dict from the LLM response.

    Returns:
        A TimelineStructure with chapters and overview.

    Raises:
        LLMError: If required keys are missing.
    """
    try:
        chapters = [
            Chapter(
                timestamp=c.get("timestamp", "00:00"),
                title=c["title"],
                summary=c.get("summary", ""),
                content=c["content"],
            )
            for c in data["chapters"]
        ]
        return TimelineStructure(chapters=chapters, overview=data.get("overview", ""))
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Failed to parse 'timeline' structure: {exc}") from exc


def _parse_cookbook(data: dict) -> CookbookStructure:
    """Parse a JSON dict into a CookbookStructure dataclass.

    Args:
        data: Parsed JSON dict from the LLM response.

    Returns:
        A CookbookStructure with prerequisites, instructions, notes, difficulty.

    Raises:
        LLMError: If required keys are missing.
    """
    try:
        return CookbookStructure(
            prerequisites=data.get("prerequisites", []),
            instructions=data["instructions"],
            notes=data.get("notes", []),
            difficulty=data.get("difficulty", "intermediate"),
        )
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Failed to parse 'cookbook' structure: {exc}") from exc


def _parse_slides(data: dict) -> SlidesStructure:
    """Parse a JSON dict into a SlidesStructure dataclass.

    Args:
        data: Parsed JSON dict from the LLM response.

    Returns:
        A SlidesStructure with slides and title_slide subtitle.

    Raises:
        LLMError: If required keys are missing.
    """
    try:
        slides = [
            Slide(heading=s["heading"], bullets=s.get("bullets", []))
            for s in data["slides"]
        ]
        return SlidesStructure(slides=slides, title_slide=data.get("title_slide", ""))
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Failed to parse 'slides' structure: {exc}") from exc


# Map style names to their parser functions.
_PARSERS: dict[str, callable] = {
    "steps": _parse_steps,
    "cornell": _parse_cornell,
    "flashcard": _parse_flashcard,
    "timeline": _parse_timeline,
    "cookbook": _parse_cookbook,
    "slides": _parse_slides,
}


# ---------------------------------------------------------------------------
# Public API — generate structured tutorial data from a transcript
# ---------------------------------------------------------------------------

def generate_tutorial(
    style: str,
    title: str,
    transcript_text: str,
    endpoint: str = "auto",
    ollama_url: str | None = None,
    *,
    sources: SourceList | None = None,
):
    """Generate a structured tutorial from one or more transcripts using an LLM.

    This is the main entry point for tutorial generation.  It:
    1. Builds the appropriate prompt for the requested style.
    2. Sends the prompt to the selected LLM backend.
    3. Parses the JSON response into a typed dataclass.

    The returned dataclass is then passed to templates.render_tutorial()
    to produce the final HTML document.

    Supports two calling conventions:
      - Single source (backward-compatible): pass title + transcript_text.
      - Multi-source: pass sources=[(title1, text1), (title2, text2), ...].
        When sources is provided, the title and transcript_text args are ignored.
        Multi-source prompts instruct the LLM to integrate overlapping content
        and keep orthogonal (unrelated) content in separate sections.

    Args:
        style:           Tutorial style name (one of TUTORIAL_STYLES).
        title:           The video's title (used for single-source prompts).
        transcript_text: The full plain-text transcript (single-source).
        endpoint:        LLM backend: "auto", "anthropic", or "ollama".
        ollama_url:      Base URL for Ollama server (optional).
        sources:         Optional list of (title, transcript_text) pairs for
                         multi-source tutorial generation.  When provided,
                         title and transcript_text args are ignored.

    Returns:
        A style-specific dataclass (StepsStructure, CornellStructure, etc.).

    Raises:
        ValueError: If style is not a recognized tutorial style.
        LLMError:   If the LLM call fails or returns unparseable output.
    """
    if style not in TUTORIAL_STYLES:
        raise ValueError(
            f"Unknown tutorial style {style!r}; "
            f"expected one of: {', '.join(TUTORIAL_STYLES)}"
        )

    # Normalize input: if explicit sources list is provided, use it;
    # otherwise wrap the single title+text into a one-element list.
    source_list: SourceList = sources if sources else [(title, transcript_text)]

    # Step 1: Build the prompts for this style.
    prompt_builder = _PROMPT_BUILDERS[style]
    system_prompt, user_prompt = prompt_builder(source_list)

    # Step 2: Call the LLM.
    raw_response = call_llm(system_prompt, user_prompt, endpoint, ollama_url)

    # Step 3: Parse the JSON response into a typed dataclass.
    data = _parse_json(raw_response)
    parser = _PARSERS[style]
    return parser(data)
