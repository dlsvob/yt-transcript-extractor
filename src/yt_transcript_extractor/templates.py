"""
templates.py — HTML templates for the 6 tutorial styles.

Each template is a self-contained HTML document with inline CSS and minimal
vanilla JS.  No external dependencies — the files render correctly when
opened directly in any modern browser.

Design philosophy:
  - Visual consistency with the existing "doc" format (Ant Design v5 tokens).
  - Each template extends the base look with style-specific layout elements.
  - Templates are Python format-strings; curly braces in CSS/JS are doubled.
  - Render functions accept typed dataclasses from tutorials.py and return
    complete HTML strings.

Template inventory:
  render_steps()     — Numbered step-by-step lesson with progress sidebar.
  render_cornell()   — Two-column Cornell notes with summary footer.
  render_flashcard() — Click-to-reveal flashcards in a responsive grid.
  render_timeline()  — Vertical timeline with chapter markers and keyboard nav.
  render_cookbook()   — Recipe card with prerequisites, instructions, notes.
  render_slides()    — Full-viewport slides with CSS scroll-snap.
"""

from __future__ import annotations

import html as html_mod

from yt_transcript_extractor.tutorials import (
    CookbookStructure,
    CornellStructure,
    FlashcardStructure,
    SlidesStructure,
    StepsStructure,
    TimelineStructure,
)


# ---------------------------------------------------------------------------
# Shared CSS — base tokens reused across all templates
# ---------------------------------------------------------------------------
# Extracted as a constant so every template shares the same visual foundation.
# These are Ant Design v5 design tokens matching the existing doc format.

_BASE_CSS = """\
  :root {
    --ant-color-bg-container: #ffffff;
    --ant-color-bg-layout: #f0f2f5;
    --ant-color-bg-elevated: #fafafa;
    --ant-color-border: #d9d9d9;
    --ant-color-text: rgba(0, 0, 0, 0.88);
    --ant-color-text-secondary: rgba(0, 0, 0, 0.65);
    --ant-color-text-tertiary: rgba(0, 0, 0, 0.45);
    --ant-color-primary: #1677ff;
    --ant-color-success: #52c41a;
    --ant-color-warning: #faad14;
    --ant-color-error: #ff4d4f;
    --ant-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                        'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
    --ant-font-size: 15px;
    --ant-line-height: 1.5714;
    --ant-border-radius: 8px;
    --ant-padding-lg: 24px;
    --ant-padding-md: 16px;
    --ant-padding-sm: 12px;
  }

  *, *::before, *::after { box-sizing: border-box; }

  body {
    font-family: var(--ant-font-family);
    font-size: var(--ant-font-size);
    line-height: var(--ant-line-height);
    color: var(--ant-color-text);
    background: linear-gradient(135deg, #f0f2f5 0%, #e8ecf1 100%);
    min-height: 100vh;
    margin: 0;
    padding: var(--ant-padding-lg);
    -webkit-font-smoothing: antialiased;
  }

  .container {
    max-width: 900px;
    margin: 0 auto;
    background: var(--ant-color-bg-container);
    border-radius: var(--ant-border-radius);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
    padding: var(--ant-padding-lg);
  }

  h1 {
    font-weight: 600;
    font-size: 30px;
    line-height: 1.2;
    margin: 0 0 8px 0;
    color: var(--ant-color-text);
  }

  .subtitle {
    color: var(--ant-color-text-tertiary);
    font-size: 14px;
    margin: 0 0 var(--ant-padding-lg) 0;
    padding-bottom: var(--ant-padding-sm);
    border-bottom: 2px solid var(--ant-color-primary);
  }

  /* --- Sources banner — shown when tutorial combines multiple videos --- */
  .sources-banner {
    background: var(--ant-color-bg-elevated);
    border: 1px solid var(--ant-color-border);
    border-radius: var(--ant-border-radius);
    padding: var(--ant-padding-sm) var(--ant-padding-md);
    margin-bottom: var(--ant-padding-lg);
  }

  .sources-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ant-color-text-tertiary);
    margin-bottom: 8px;
  }

  .sources-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .source-pill {
    font-size: 13px;
    background: rgba(22, 119, 255, 0.06);
    color: var(--ant-color-primary);
    padding: 4px 12px;
    border-radius: 12px;
    border: 1px solid rgba(22, 119, 255, 0.12);
  }
"""


def _esc(text: str) -> str:
    """HTML-escape a string to prevent XSS from transcript content.

    All user-supplied text (titles, transcript content, LLM output) passes
    through this before being interpolated into HTML templates.

    Args:
        text: Raw string to escape.

    Returns:
        HTML-escaped string safe for embedding in HTML.
    """
    return html_mod.escape(text, quote=True)


def _render_sources_banner(source_titles: list[str] | None) -> str:
    """Render a source attribution banner for multi-source tutorials.

    When a tutorial is generated from multiple video transcripts, this
    produces a styled list of source video titles so the reader knows
    which videos contributed to the content.

    Returns empty string for single-source tutorials (no banner needed).

    Args:
        source_titles: List of video titles that were combined, or None
                       for single-source tutorials.

    Returns:
        HTML string for the sources banner, or empty string.
    """
    if not source_titles or len(source_titles) <= 1:
        return ""

    pills = "".join(
        f'<span class="source-pill">{_esc(t)}</span>'
        for t in source_titles
    )
    return (
        f'<div class="sources-banner">'
        f'<span class="sources-label">Combined from {len(source_titles)} sources:</span>'
        f'<div class="sources-list">{pills}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Template 1: Steps — step-by-step lesson
# ---------------------------------------------------------------------------

_STEPS_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Step-by-Step</title>
<style>
{base_css}

  /* --- Steps-specific styles --- */
  .summary {{
    background: rgba(22, 119, 255, 0.04);
    border-left: 3px solid var(--ant-color-primary);
    padding: var(--ant-padding-md);
    margin-bottom: var(--ant-padding-lg);
    border-radius: 0 var(--ant-border-radius) var(--ant-border-radius) 0;
    color: var(--ant-color-text-secondary);
  }}

  .step {{
    position: relative;
    padding: var(--ant-padding-md);
    padding-left: 60px;
    margin-bottom: var(--ant-padding-md);
    border: 1px solid var(--ant-color-border);
    border-radius: var(--ant-border-radius);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
  }}

  .step:hover {{
    box-shadow: 0 2px 8px rgba(22, 119, 255, 0.1);
    border-color: rgba(22, 119, 255, 0.3);
  }}

  /* Circle with the step number, positioned to the left of the card. */
  .step-number {{
    position: absolute;
    left: 14px;
    top: 14px;
    width: 32px;
    height: 32px;
    background: var(--ant-color-primary);
    color: #fff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
  }}

  .step h3 {{
    margin: 0 0 8px 0;
    font-size: 17px;
    font-weight: 600;
  }}

  /* Key concepts rendered as small pill badges below the step title. */
  .concepts {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
  }}

  .concept-pill {{
    font-size: 12px;
    background: rgba(22, 119, 255, 0.08);
    color: var(--ant-color-primary);
    padding: 2px 10px;
    border-radius: 12px;
    border: 1px solid rgba(22, 119, 255, 0.15);
  }}

  .step-content {{
    color: var(--ant-color-text-secondary);
    line-height: 1.8;
  }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">Step-by-Step Tutorial</p>
{summary_html}
{steps_html}
</div>
</body>
</html>"""


def render_steps(structure: StepsStructure, title: str, source_titles: list[str] | None = None) -> str:
    """Render a StepsStructure as a complete HTML document.

    Produces a numbered step-by-step tutorial layout.  Each step is a card
    with a circled number, title, concept pills, and detailed content.

    Args:
        structure:     The StepsStructure dataclass from the LLM.
        title:         The video title (used in <title> and <h1>).
        source_titles: Optional list of source video titles for attribution.

    Returns:
        A complete HTML string ready to write to a file.
    """
    # Sources banner for multi-video tutorials.
    sources_html = _render_sources_banner(source_titles)

    # Build the summary block (shown at the top if the LLM provided one).
    summary_html = ""
    if structure.summary:
        summary_html = f'<div class="summary">{_esc(structure.summary)}</div>'

    # Build each step card.
    step_cards = []
    for step in structure.steps:
        # Concept pills
        concepts = "".join(
            f'<span class="concept-pill">{_esc(c)}</span>'
            for c in step.key_concepts
        )
        concepts_html = f'<div class="concepts">{concepts}</div>' if concepts else ""

        card = (
            f'<div class="step">'
            f'<div class="step-number">{step.number}</div>'
            f'<h3>{_esc(step.title)}</h3>'
            f'{concepts_html}'
            f'<div class="step-content">{_esc(step.content)}</div>'
            f'</div>'
        )
        step_cards.append(card)

    steps_html = "\n".join(step_cards)

    return _STEPS_TEMPLATE.format(
        title=_esc(title),
        base_css=_BASE_CSS,
        summary_html=sources_html + summary_html,
        steps_html=steps_html,
    )


# ---------------------------------------------------------------------------
# Template 2: Cornell — two-column notes
# ---------------------------------------------------------------------------

_CORNELL_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Cornell Notes</title>
<style>
{base_css}

  /* --- Cornell-specific styles --- */
  /* Two-column table layout mimicking a real Cornell notes page. */
  .cornell-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border: 1px solid var(--ant-color-border);
    border-radius: var(--ant-border-radius);
    overflow: hidden;
    margin-bottom: var(--ant-padding-lg);
  }}

  /* Header row labeling the two columns. */
  .cornell-table thead th {{
    background: var(--ant-color-bg-elevated);
    padding: 10px var(--ant-padding-md);
    font-weight: 600;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ant-color-text-tertiary);
    border-bottom: 2px solid var(--ant-color-primary);
  }}

  .cornell-table thead th:first-child {{
    width: 25%;
    text-align: left;
  }}

  .cornell-table tbody td {{
    padding: var(--ant-padding-md);
    vertical-align: top;
    border-bottom: 1px solid var(--ant-color-border);
  }}

  .cornell-table tbody tr:last-child td {{
    border-bottom: none;
  }}

  /* Cue column — bold keywords/questions in a slightly tinted background. */
  .cue-cell {{
    background: rgba(22, 119, 255, 0.03);
    font-weight: 600;
    color: var(--ant-color-primary);
    border-right: 2px solid var(--ant-color-primary);
  }}

  /* Notes column — regular text. */
  .notes-cell {{
    color: var(--ant-color-text-secondary);
    line-height: 1.8;
  }}

  /* Summary box at the bottom of the page. */
  .cornell-summary {{
    background: var(--ant-color-bg-elevated);
    border: 1px solid var(--ant-color-border);
    border-radius: var(--ant-border-radius);
    padding: var(--ant-padding-md);
  }}

  .cornell-summary h2 {{
    font-size: 15px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ant-color-text-tertiary);
    margin: 0 0 8px 0;
  }}

  .cornell-summary p {{
    margin: 0;
    color: var(--ant-color-text-secondary);
    line-height: 1.8;
  }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">Cornell Notes</p>
<table class="cornell-table">
<thead><tr><th>Cues</th><th>Notes</th></tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
{summary_html}
</div>
</body>
</html>"""


def render_cornell(structure: CornellStructure, title: str, source_titles: list[str] | None = None) -> str:
    """Render a CornellStructure as a complete HTML document.

    Produces a two-column Cornell notes layout with a summary footer.

    Args:
        structure:     The CornellStructure dataclass from the LLM.
        title:         The video title.
        source_titles: Optional list of source video titles for attribution.

    Returns:
        A complete HTML string.
    """
    rows = []
    for note in structure.notes:
        rows.append(
            f'<tr>'
            f'<td class="cue-cell">{_esc(note.cue)}</td>'
            f'<td class="notes-cell">{_esc(note.notes)}</td>'
            f'</tr>'
        )
    rows_html = "\n".join(rows)

    summary_html = ""
    if structure.summary:
        summary_html = (
            f'<div class="cornell-summary">'
            f'<h2>Summary</h2>'
            f'<p>{_esc(structure.summary)}</p>'
            f'</div>'
        )

    # Inject sources banner between subtitle and table via the rows_html slot.
    sources_html = _render_sources_banner(source_titles)

    return _CORNELL_TEMPLATE.format(
        title=_esc(title),
        base_css=_BASE_CSS,
        rows_html=rows_html,
        summary_html=sources_html + summary_html,
    )


# ---------------------------------------------------------------------------
# Template 3: Flashcard — click-to-reveal Q&A cards
# ---------------------------------------------------------------------------

_FLASHCARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Flashcards</title>
<style>
{base_css}

  .container {{ max-width: 1000px; }}

  .topic-label {{
    display: inline-block;
    background: rgba(22, 119, 255, 0.08);
    color: var(--ant-color-primary);
    font-size: 13px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 12px;
    margin-bottom: var(--ant-padding-lg);
  }}

  /* Responsive grid of flashcard tiles. */
  .card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: var(--ant-padding-md);
  }}

  /* Each card is a clickable container with a front (question) and
     back (answer) that toggles on click via a JS class flip. */
  .flashcard {{
    perspective: 800px;
    cursor: pointer;
    min-height: 200px;
  }}

  .flashcard-inner {{
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 200px;
    transition: transform 0.5s ease;
    transform-style: preserve-3d;
  }}

  .flashcard.flipped .flashcard-inner {{
    transform: rotateY(180deg);
  }}

  .card-front, .card-back {{
    position: absolute;
    inset: 0;
    backface-visibility: hidden;
    border-radius: var(--ant-border-radius);
    border: 1px solid var(--ant-color-border);
    padding: var(--ant-padding-lg);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }}

  .card-front {{
    background: var(--ant-color-bg-container);
  }}

  .card-front::after {{
    content: "click to reveal";
    position: absolute;
    bottom: 10px;
    font-size: 11px;
    color: var(--ant-color-text-tertiary);
  }}

  .card-front p {{
    font-weight: 600;
    font-size: 16px;
    margin: 0;
  }}

  .card-back {{
    background: rgba(22, 119, 255, 0.04);
    border-color: rgba(22, 119, 255, 0.2);
    transform: rotateY(180deg);
  }}

  .card-back p {{
    margin: 0;
    color: var(--ant-color-text-secondary);
    line-height: 1.7;
  }}

  /* Card counter badge in the top-right corner. */
  .card-num {{
    position: absolute;
    top: 8px;
    right: 10px;
    font-size: 11px;
    color: var(--ant-color-text-tertiary);
  }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">Study Flashcards</p>
{topic_html}
<div class="card-grid">
{cards_html}
</div>
</div>
<script>
/* Toggle the 'flipped' class on click to reveal/hide the answer. */
document.querySelectorAll('.flashcard').forEach(function(card) {{
  card.addEventListener('click', function() {{
    this.classList.toggle('flipped');
  }});
}});
</script>
</body>
</html>"""


def render_flashcard(structure: FlashcardStructure, title: str, source_titles: list[str] | None = None) -> str:
    """Render a FlashcardStructure as a complete HTML document.

    Produces a responsive grid of flip-on-click flashcards.

    Args:
        structure:     The FlashcardStructure dataclass from the LLM.
        title:         The video title.
        source_titles: Optional list of source video titles for attribution.

    Returns:
        A complete HTML string.
    """
    sources_html = _render_sources_banner(source_titles)

    topic_html = ""
    if structure.topic:
        topic_html = f'<span class="topic-label">{_esc(structure.topic)}</span>'

    cards = []
    for i, card in enumerate(structure.cards, 1):
        cards.append(
            f'<div class="flashcard">'
            f'<div class="flashcard-inner">'
            f'<div class="card-front">'
            f'<span class="card-num">{i}/{len(structure.cards)}</span>'
            f'<p>{_esc(card.question)}</p>'
            f'</div>'
            f'<div class="card-back">'
            f'<span class="card-num">{i}/{len(structure.cards)}</span>'
            f'<p>{_esc(card.answer)}</p>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
    cards_html = "\n".join(cards)

    return _FLASHCARD_TEMPLATE.format(
        title=_esc(title),
        base_css=_BASE_CSS,
        topic_html=sources_html + topic_html,
        cards_html=cards_html,
    )


# ---------------------------------------------------------------------------
# Template 4: Timeline — vertical chapters with keyboard nav
# ---------------------------------------------------------------------------

_TIMELINE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Timeline</title>
<style>
{base_css}

  .overview {{
    color: var(--ant-color-text-secondary);
    margin-bottom: var(--ant-padding-lg);
    line-height: 1.7;
  }}

  .kbd-hint {{
    font-size: 12px;
    color: var(--ant-color-text-tertiary);
    margin-bottom: var(--ant-padding-lg);
  }}

  .kbd-hint kbd {{
    display: inline-block;
    padding: 2px 6px;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 11px;
    background: var(--ant-color-bg-elevated);
    border: 1px solid var(--ant-color-border);
    border-radius: 4px;
    box-shadow: 0 1px 0 var(--ant-color-border);
  }}

  /* Vertical timeline — a line on the left with chapter nodes. */
  .timeline {{
    position: relative;
    padding-left: 40px;
  }}

  /* The vertical connecting line. */
  .timeline::before {{
    content: "";
    position: absolute;
    left: 15px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--ant-color-border);
  }}

  .chapter {{
    position: relative;
    margin-bottom: var(--ant-padding-lg);
    padding: var(--ant-padding-md);
    border: 1px solid var(--ant-color-border);
    border-radius: var(--ant-border-radius);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }}

  /* Highlight the focused/active chapter for keyboard navigation. */
  .chapter.active {{
    border-color: var(--ant-color-primary);
    box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.15);
  }}

  /* Circle node on the timeline line. */
  .chapter::before {{
    content: "";
    position: absolute;
    left: -33px;
    top: 18px;
    width: 10px;
    height: 10px;
    background: var(--ant-color-bg-container);
    border: 2px solid var(--ant-color-primary);
    border-radius: 50%;
  }}

  .chapter.active::before {{
    background: var(--ant-color-primary);
  }}

  .chapter-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }}

  /* Timestamp badge next to the chapter title. */
  .chapter-ts {{
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 12px;
    font-weight: 500;
    color: var(--ant-color-primary);
    background: rgba(22, 119, 255, 0.1);
    border: 1px solid rgba(22, 119, 255, 0.18);
    padding: 2px 8px;
    border-radius: 4px;
    flex-shrink: 0;
  }}

  .chapter-title {{
    font-weight: 600;
    font-size: 16px;
    margin: 0;
  }}

  .chapter-summary {{
    color: var(--ant-color-text-tertiary);
    font-size: 13px;
    font-style: italic;
    margin-bottom: 8px;
  }}

  .chapter-content {{
    color: var(--ant-color-text-secondary);
    line-height: 1.8;
  }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">Timeline &amp; Chapters</p>
{overview_html}
<p class="kbd-hint">Navigate: <kbd>j</kbd> / <kbd>k</kbd> or <kbd>&darr;</kbd> / <kbd>&uarr;</kbd></p>
<div class="timeline">
{chapters_html}
</div>
</div>
<script>
/* Keyboard navigation: j/Down moves to next chapter, k/Up to previous. */
(function() {{
  var chapters = document.querySelectorAll('.chapter');
  var current = 0;
  if (chapters.length > 0) chapters[0].classList.add('active');

  document.addEventListener('keydown', function(e) {{
    if (e.key === 'j' || e.key === 'ArrowDown') {{
      e.preventDefault();
      if (current < chapters.length - 1) {{
        chapters[current].classList.remove('active');
        current++;
        chapters[current].classList.add('active');
        chapters[current].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      }}
    }} else if (e.key === 'k' || e.key === 'ArrowUp') {{
      e.preventDefault();
      if (current > 0) {{
        chapters[current].classList.remove('active');
        current--;
        chapters[current].classList.add('active');
        chapters[current].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""


def render_timeline(structure: TimelineStructure, title: str, source_titles: list[str] | None = None) -> str:
    """Render a TimelineStructure as a complete HTML document.

    Produces a vertical timeline with chapter nodes and keyboard navigation.

    Args:
        structure:     The TimelineStructure dataclass from the LLM.
        title:         The video title.
        source_titles: Optional list of source video titles for attribution.

    Returns:
        A complete HTML string.
    """
    sources_html = _render_sources_banner(source_titles)

    overview_html = ""
    if structure.overview:
        overview_html = f'<p class="overview">{_esc(structure.overview)}</p>'

    chapters = []
    for ch in structure.chapters:
        summary_line = ""
        if ch.summary:
            summary_line = f'<div class="chapter-summary">{_esc(ch.summary)}</div>'

        chapters.append(
            f'<div class="chapter">'
            f'<div class="chapter-header">'
            f'<span class="chapter-ts">{_esc(ch.timestamp)}</span>'
            f'<h3 class="chapter-title">{_esc(ch.title)}</h3>'
            f'</div>'
            f'{summary_line}'
            f'<div class="chapter-content">{_esc(ch.content)}</div>'
            f'</div>'
        )
    chapters_html = "\n".join(chapters)

    return _TIMELINE_TEMPLATE.format(
        title=_esc(title),
        base_css=_BASE_CSS,
        overview_html=sources_html + overview_html,
        chapters_html=chapters_html,
    )


# ---------------------------------------------------------------------------
# Template 5: Cookbook — recipe card
# ---------------------------------------------------------------------------

_COOKBOOK_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Cookbook</title>
<style>
{base_css}

  /* Difficulty badge next to the subtitle. */
  .difficulty {{
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 3px 10px;
    border-radius: 12px;
    margin-left: 8px;
  }}

  .difficulty-beginner {{
    background: rgba(82, 196, 26, 0.1);
    color: var(--ant-color-success);
    border: 1px solid rgba(82, 196, 26, 0.2);
  }}

  .difficulty-intermediate {{
    background: rgba(250, 173, 20, 0.1);
    color: var(--ant-color-warning);
    border: 1px solid rgba(250, 173, 20, 0.2);
  }}

  .difficulty-advanced {{
    background: rgba(255, 77, 79, 0.1);
    color: var(--ant-color-error);
    border: 1px solid rgba(255, 77, 79, 0.2);
  }}

  /* Section headings within the recipe card. */
  .section-heading {{
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ant-color-text-tertiary);
    margin: var(--ant-padding-lg) 0 var(--ant-padding-sm) 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--ant-color-border);
  }}

  .section-heading:first-of-type {{
    margin-top: 0;
  }}

  /* Prerequisites shown as a horizontal pill list. */
  .prereq-list {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    list-style: none;
    padding: 0;
    margin: 0;
  }}

  .prereq-list li {{
    background: var(--ant-color-bg-elevated);
    border: 1px solid var(--ant-color-border);
    padding: 6px 14px;
    border-radius: var(--ant-border-radius);
    font-size: 14px;
  }}

  /* Numbered instruction steps. */
  .instructions {{
    counter-reset: instruction;
    list-style: none;
    padding: 0;
    margin: 0;
  }}

  .instructions li {{
    counter-increment: instruction;
    position: relative;
    padding: var(--ant-padding-sm) 0 var(--ant-padding-sm) 44px;
    border-bottom: 1px solid var(--ant-color-border);
    line-height: 1.7;
    color: var(--ant-color-text-secondary);
  }}

  .instructions li:last-child {{
    border-bottom: none;
  }}

  /* Auto-numbered circle before each instruction. */
  .instructions li::before {{
    content: counter(instruction);
    position: absolute;
    left: 0;
    top: 12px;
    width: 28px;
    height: 28px;
    background: var(--ant-color-primary);
    color: #fff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 13px;
  }}

  /* Notes shown as a list with a lightbulb-style left border. */
  .notes-list {{
    list-style: none;
    padding: 0;
    margin: 0;
  }}

  .notes-list li {{
    padding: 8px 0 8px var(--ant-padding-md);
    border-left: 3px solid var(--ant-color-warning);
    margin-bottom: 8px;
    color: var(--ant-color-text-secondary);
    line-height: 1.7;
    background: rgba(250, 173, 20, 0.03);
    border-radius: 0 var(--ant-border-radius) var(--ant-border-radius) 0;
  }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">Cookbook / How-To{difficulty_badge}</p>
{prereq_html}
{instructions_html}
{notes_html}
</div>
</body>
</html>"""


def render_cookbook(structure: CookbookStructure, title: str, source_titles: list[str] | None = None) -> str:
    """Render a CookbookStructure as a complete HTML document.

    Produces a recipe-card layout with prerequisites, numbered instructions,
    and tips/notes sections.

    Args:
        structure:     The CookbookStructure dataclass from the LLM.
        title:         The video title.
        source_titles: Optional list of source video titles for attribution.

    Returns:
        A complete HTML string.
    """
    # Difficulty badge — color-coded by level.
    diff = structure.difficulty.lower()
    diff_class = f"difficulty-{diff}" if diff in ("beginner", "intermediate", "advanced") else "difficulty-intermediate"
    difficulty_badge = f' <span class="difficulty {diff_class}">{_esc(structure.difficulty)}</span>'

    sources_html = _render_sources_banner(source_titles)

    # Prerequisites section.
    prereq_html = ""
    if structure.prerequisites:
        items = "".join(f"<li>{_esc(p)}</li>" for p in structure.prerequisites)
        prereq_html = (
            f'<h2 class="section-heading">Prerequisites</h2>'
            f'<ul class="prereq-list">{items}</ul>'
        )

    # Instructions section.
    items = "".join(f"<li>{_esc(inst)}</li>" for inst in structure.instructions)
    instructions_html = (
        f'<h2 class="section-heading">Instructions</h2>'
        f'<ol class="instructions">{items}</ol>'
    )

    # Notes section.
    notes_html = ""
    if structure.notes:
        items = "".join(f"<li>{_esc(n)}</li>" for n in structure.notes)
        notes_html = (
            f'<h2 class="section-heading">Notes &amp; Tips</h2>'
            f'<ul class="notes-list">{items}</ul>'
        )

    return _COOKBOOK_TEMPLATE.format(
        title=_esc(title),
        base_css=_BASE_CSS,
        difficulty_badge=difficulty_badge,
        prereq_html=sources_html + prereq_html,
        instructions_html=instructions_html,
        notes_html=notes_html,
    )


# ---------------------------------------------------------------------------
# Template 6: Slides — full-viewport scroll-snap presentation
# ---------------------------------------------------------------------------

_SLIDES_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Slides</title>
<style>
  /* --- Slides override the base layout entirely for full-viewport. --- */
  :root {{
    --ant-color-bg-container: #ffffff;
    --ant-color-border: #d9d9d9;
    --ant-color-text: rgba(0, 0, 0, 0.88);
    --ant-color-text-secondary: rgba(0, 0, 0, 0.65);
    --ant-color-text-tertiary: rgba(0, 0, 0, 0.45);
    --ant-color-primary: #1677ff;
    --ant-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                        'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
    --ant-border-radius: 8px;
  }}

  *, *::before, *::after {{ box-sizing: border-box; }}

  html {{
    scroll-snap-type: y mandatory;
    overflow-y: scroll;
    height: 100%;
  }}

  body {{
    font-family: var(--ant-font-family);
    margin: 0;
    padding: 0;
    height: 100%;
    -webkit-font-smoothing: antialiased;
  }}

  /* Each slide is a full-viewport section with scroll-snap alignment. */
  .slide {{
    min-height: 100vh;
    scroll-snap-align: start;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 60px 40px;
    position: relative;
  }}

  /* Alternate slide backgrounds for visual rhythm. */
  .slide:nth-child(odd) {{
    background: linear-gradient(135deg, #f0f2f5 0%, #e8ecf1 100%);
  }}

  .slide:nth-child(even) {{
    background: var(--ant-color-bg-container);
  }}

  /* Title slide — larger, centered, with accent color. */
  .slide-title {{
    text-align: center;
  }}

  .slide-title h1 {{
    font-size: 48px;
    font-weight: 700;
    margin: 0 0 16px 0;
    color: var(--ant-color-text);
  }}

  .slide-title p {{
    font-size: 20px;
    color: var(--ant-color-text-tertiary);
    margin: 0;
  }}

  /* Content slides. */
  .slide-content {{
    max-width: 800px;
    width: 100%;
  }}

  .slide-content h2 {{
    font-size: 32px;
    font-weight: 600;
    margin: 0 0 24px 0;
    color: var(--ant-color-text);
    padding-bottom: 12px;
    border-bottom: 3px solid var(--ant-color-primary);
  }}

  .slide-content ul {{
    list-style: none;
    padding: 0;
    margin: 0;
  }}

  .slide-content li {{
    position: relative;
    padding: 12px 0 12px 28px;
    font-size: 18px;
    line-height: 1.6;
    color: var(--ant-color-text-secondary);
  }}

  /* Custom bullet — a small primary-colored dot. */
  .slide-content li::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 20px;
    width: 8px;
    height: 8px;
    background: var(--ant-color-primary);
    border-radius: 50%;
  }}

  /* Slide counter in the bottom-right corner. */
  .slide-num {{
    position: absolute;
    bottom: 16px;
    right: 24px;
    font-size: 13px;
    color: var(--ant-color-text-tertiary);
  }}

  /* Navigation hint at the bottom of the title slide. */
  .nav-hint {{
    position: absolute;
    bottom: 30px;
    font-size: 13px;
    color: var(--ant-color-text-tertiary);
    animation: bounce 2s ease infinite;
  }}

  @keyframes bounce {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(6px); }}
  }}

  /* Print layout: one slide per page. */
  @media print {{
    .slide {{
      min-height: 100vh;
      page-break-after: always;
      break-after: page;
    }}
    .nav-hint {{ display: none; }}
  }}
</style>
</head>
<body>
{slides_html}
</body>
</html>"""


def render_slides(structure: SlidesStructure, title: str, source_titles: list[str] | None = None) -> str:
    """Render a SlidesStructure as a complete HTML document.

    Produces a full-viewport slide deck with CSS scroll-snap navigation.
    Each slide covers one topic with a heading and bullet points.
    Includes a title slide at the beginning and print-friendly styles.

    Args:
        structure:     The SlidesStructure dataclass from the LLM.
        title:         The video title.
        source_titles: Optional list of source video titles for attribution.

    Returns:
        A complete HTML string.
    """
    total = len(structure.slides) + 1  # +1 for title slide

    # For slides, source attribution goes on the title slide as a subtitle line.
    sources_line = ""
    if source_titles and len(source_titles) > 1:
        sources_line = (
            f'<p style="font-size:14px; color:rgba(0,0,0,0.45); margin-top:12px;">'
            f'Combined from: {", ".join(_esc(t) for t in source_titles)}'
            f'</p>'
        )

    # Title slide.
    slides = [
        f'<div class="slide slide-title">'
        f'<h1>{_esc(title)}</h1>'
        f'<p>{_esc(structure.title_slide)}</p>'
        f'{sources_line}'
        f'<span class="slide-num">1 / {total}</span>'
        f'<span class="nav-hint">Scroll to begin &darr;</span>'
        f'</div>'
    ]

    # Content slides.
    for i, slide in enumerate(structure.slides, 2):
        bullets = "".join(f"<li>{_esc(b)}</li>" for b in slide.bullets)
        slides.append(
            f'<div class="slide">'
            f'<div class="slide-content">'
            f'<h2>{_esc(slide.heading)}</h2>'
            f'<ul>{bullets}</ul>'
            f'</div>'
            f'<span class="slide-num">{i} / {total}</span>'
            f'</div>'
        )

    slides_html = "\n".join(slides)

    return _SLIDES_TEMPLATE.format(
        title=_esc(title),
        slides_html=slides_html,
    )


# ---------------------------------------------------------------------------
# Public API — dispatch to the right renderer
# ---------------------------------------------------------------------------

# Map style names to their render functions.
_RENDERERS = {
    "steps": render_steps,
    "cornell": render_cornell,
    "flashcard": render_flashcard,
    "timeline": render_timeline,
    "cookbook": render_cookbook,
    "slides": render_slides,
}


def render_tutorial(style: str, structure, title: str, source_titles: list[str] | None = None) -> str:
    """Dispatch to the appropriate template renderer.

    This is the main entry point for template rendering.  It takes the
    structured data from tutorials.generate_tutorial() and the style name,
    and returns a complete HTML document string.

    Args:
        style:         Tutorial style name (one of tutorials.TUTORIAL_STYLES).
        structure:     The style-specific dataclass from generate_tutorial().
        title:         The video title (used in headings and <title>).
        source_titles: Optional list of source video titles for multi-source
                       attribution.  When provided and length > 1, a sources
                       banner is rendered in the template header.

    Returns:
        A complete HTML string ready to write to a file.

    Raises:
        ValueError: If style is not a recognized tutorial style.
    """
    renderer = _RENDERERS.get(style)
    if renderer is None:
        raise ValueError(
            f"Unknown tutorial style {style!r}; "
            f"expected one of: {', '.join(_RENDERERS.keys())}"
        )
    return renderer(structure, title, source_titles=source_titles)
