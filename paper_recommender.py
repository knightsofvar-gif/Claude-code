#!/usr/bin/env python3
"""
Paper Recommendation Generator

Generates:
  1. Foundational/seminal papers for a research topic (Claude with deep reasoning)
  2. Newly released papers for a research topic (Claude with web search)
"""

import sys
import json
import argparse
import textwrap

import anthropic


# ─── System prompts ───────────────────────────────────────────────────────────

FOUNDATIONAL_SYSTEM = """\
You are an expert research librarian and academic advisor specialised in
recommending scientific literature. You respond ONLY with valid JSON and
nothing else — no markdown fences, no prose, no code blocks.\
"""

FOUNDATIONAL_PROMPT = """\
List the {n} most important foundational/seminal papers for the research topic:
"{topic}"

Return a JSON array where every element has these exact keys:
  "title"      – full paper title
  "authors"    – list of author strings  (e.g. ["Vaswani, Ashish", "Shazeer, Noam"])
  "year"       – publication year as an integer
  "venue"      – journal or conference name
  "why"        – 1–2 sentence explanation of why this paper is foundational
  "url"        – best available URL (DOI preferred, arXiv abs URL if no DOI)

Criteria — prefer papers that:
  • Introduced a key method, concept, architecture, or dataset
  • Are highly cited within the field
  • Are still actively referenced in current research

Return ONLY the JSON array — no markdown, no extra text.\
"""

RECENT_SYSTEM = """\
You are a research assistant that finds the most relevant newly published papers
on a given topic. You use web search to locate the latest academic papers.
You respond ONLY with valid JSON — no markdown fences, no prose.\
"""

RECENT_PROMPT = """\
Find the {n} most relevant and high-quality papers on the topic "{topic}"
published in the last {days} days.

Search academic sources such as arXiv, Semantic Scholar, Papers With Code,
Google Scholar, and major conference/journal websites.

Return a JSON array where every element has:
  "title"      – full paper title
  "authors"    – list of author strings
  "published"  – publication or preprint date (YYYY-MM-DD or YYYY-MM)
  "venue"      – arXiv, conference, journal name (or "preprint" if unknown)
  "abstract"   – 2–3 sentence summary of the paper's contribution
  "url"        – direct link to the paper (arXiv abs URL preferred)

Return ONLY the JSON array — no markdown, no extra text.\
"""


# ─── Claude calls ─────────────────────────────────────────────────────────────

def _parse_json_response(response: anthropic.types.Message, label: str) -> list[dict]:
    """Extract and parse the JSON list from a Claude response."""
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text.strip()
            break

    # Strip accidental markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"[warning] Could not parse {label} response as JSON: {exc}", file=sys.stderr)
        print(f"[debug] Raw text:\n{text[:500]}", file=sys.stderr)
        return []


def get_foundational_papers(
    topic: str,
    n: int,
    client: anthropic.Anthropic,
) -> list[dict]:
    """Ask Claude (with adaptive thinking) for the top-n foundational papers."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=FOUNDATIONAL_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": FOUNDATIONAL_PROMPT.format(topic=topic, n=n),
            }
        ],
    )
    return _parse_json_response(response, "foundational")


def get_recent_papers(
    topic: str,
    n: int,
    days: int,
    client: anthropic.Anthropic,
) -> list[dict]:
    """Use Claude + web_search to find newly published papers."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=RECENT_SYSTEM,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": RECENT_PROMPT.format(topic=topic, n=n, days=days),
            }
        ],
    )
    return _parse_json_response(response, "recent")


# ─── Formatting ───────────────────────────────────────────────────────────────

DIVIDER  = "─" * 72
HDIVIDER = "═" * 72


def _wrap(text: str, indent: int = 4, width: int = 72) -> str:
    prefix = " " * indent
    return textwrap.fill(
        text, width=width, initial_indent=prefix, subsequent_indent=prefix
    )


def _print_header(title: str) -> None:
    print(f"\n{HDIVIDER}")
    print(f"  {title}")
    print(HDIVIDER)


def print_foundational(papers: list[dict]) -> None:
    _print_header("FOUNDATIONAL PAPERS")
    if not papers:
        print("  (no results returned)")
        return
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors") or [])
        if len(authors) > 80:
            authors = authors[:77] + "…"
        print(f"\n  [{i}] {p.get('title', 'Untitled')}")
        print(f"      {authors} ({p.get('year', '?')})")
        venue = p.get("venue", "")
        if venue:
            print(f"      {venue}")
        why = p.get("why", "")
        if why:
            print(_wrap(why))
        url = p.get("url", "")
        if url:
            print(f"    → {url}")
        print(f"  {DIVIDER}")


def print_recent(papers: list[dict]) -> None:
    _print_header("RECENTLY PUBLISHED PAPERS")
    if not papers:
        print("  No recent papers found.")
        return
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors") or [])
        if len(authors) > 80:
            authors = authors[:77] + "…"
        print(f"\n  [{i}] {p.get('title', 'Untitled')}")
        print(f"      {authors}")
        print(f"      Published: {p.get('published', 'unknown')}  |  {p.get('venue', '')}")
        abstract = p.get("abstract", "")
        if abstract:
            print(_wrap(abstract))
        url = p.get("url", "")
        if url:
            print(f"    → {url}")
        print(f"  {DIVIDER}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper recommendations for a research topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python paper_recommender.py "transformer attention mechanisms"
              python paper_recommender.py "graph neural networks" -f 10 -r 15
              python paper_recommender.py "diffusion models" --days 180
        """),
    )
    parser.add_argument("topic", help="Research topic to search for")
    parser.add_argument(
        "--foundational", "-f",
        type=int, default=8, metavar="N",
        help="Number of foundational papers to return (default: 8)",
    )
    parser.add_argument(
        "--recent", "-r",
        type=int, default=10, metavar="N",
        help="Number of recent papers to return (default: 10)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int, default=365, metavar="DAYS",
        help="How many days back to search for recent papers (default: 365)",
    )
    args = parser.parse_args()

    topic = args.topic.strip()
    print(f"\nSearching paper recommendations for: '{topic}'\n")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    print("  [1/2] Asking Claude for foundational papers (deep reasoning)…")
    foundational = get_foundational_papers(topic, args.foundational, client)

    print("  [2/2] Searching the web for recently published papers…")
    recent = get_recent_papers(topic, args.recent, args.days, client)

    print_foundational(foundational)
    print_recent(recent)
    print()


if __name__ == "__main__":
    main()
