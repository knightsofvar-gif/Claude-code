#!/usr/bin/env python3
"""
Paper Recommendation Generator

Generates:
  1. Foundational/seminal papers for a research topic (Claude with deep reasoning)
  2. Newly released papers for a research topic (Claude with web search)

Features:
  - Both API calls run concurrently (ThreadPoolExecutor)
  - Pydantic validation of every paper Claude returns
  - Disk cache in ~/.paper_recommender_cache/
      foundational papers: 30-day TTL (they don't change)
      recent papers:        1-day TTL (use --no-cache to force refresh)
"""

import sys
import json
import hashlib
import argparse
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import anthropic
from pydantic import BaseModel, ValidationError


# ─── Cache ────────────────────────────────────────────────────────────────────

CACHE_DIR             = Path.home() / ".paper_recommender_cache"
FOUNDATIONAL_TTL_DAYS = 30   # foundational papers are stable
RECENT_TTL_DAYS       = 1    # recent paper lists change daily


def _cache_key(*parts: str) -> str:
    """Stable SHA-256 key from an ordered list of strings."""
    raw = "|".join(str(p).lower().strip() for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_load(key: str, ttl_days: int) -> list[dict] | None:
    """Return cached paper list if it exists and is within TTL, else None."""
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        age_days = (datetime.now(timezone.utc) - cached_at).days
        if age_days > ttl_days:
            return None
        return data["papers"]
    except Exception:
        return None  # corrupt cache entry — treat as miss


def cache_save(key: str, papers: list[dict]) -> None:
    """Write paper list to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "papers": papers,
    }, indent=2))


# ─── Pydantic models ──────────────────────────────────────────────────────────

class FoundationalPaper(BaseModel):
    title:   str
    authors: list[str]
    year:    int
    venue:   str  = ""
    why:     str  = ""
    url:     str  = ""


class RecentPaper(BaseModel):
    title:     str
    authors:   list[str]
    published: str
    venue:     str = ""
    abstract:  str = ""
    url:       str = ""


def _validate(raw: list[dict], model: type) -> list:
    """Validate a list of dicts against a Pydantic model, skipping bad entries."""
    valid = []
    for item in raw:
        try:
            valid.append(model(**item))
        except ValidationError as exc:
            print(f"[warning] Dropping invalid entry: {exc.errors()[0]['msg']}", file=sys.stderr)
    return valid


# ─── Prompts ──────────────────────────────────────────────────────────────────

FOUNDATIONAL_SYSTEM = """\
You are an expert research librarian and academic advisor specialised in
recommending scientific literature. You respond ONLY with valid JSON —
no markdown fences, no prose, no code blocks.\
"""

FOUNDATIONAL_PROMPT = """\
List the {n} most important foundational/seminal papers for the research topic:
"{topic}"

Return a JSON array where every element has these exact keys:
  "title"   – full paper title
  "authors" – list of author strings (e.g. ["Vaswani, Ashish", "Shazeer, Noam"])
  "year"    – publication year as an integer
  "venue"   – journal or conference name
  "why"     – 1–2 sentence explanation of why this paper is foundational
  "url"     – best available URL (DOI preferred, arXiv abs URL if no DOI)

Prefer papers that introduced a key method/concept/dataset, are highly cited,
and are still actively referenced. Return ONLY the JSON array.\
"""

RECENT_SYSTEM = """\
You are a research assistant that finds the most relevant newly published papers
on a given topic using web search. You respond ONLY with valid JSON —
no markdown fences, no prose.\
"""

RECENT_PROMPT = """\
Find the {n} most relevant and high-quality papers on the topic "{topic}"
published in the last {days} days.

Search arXiv, Semantic Scholar, Papers With Code, and major conference sites.

Return a JSON array where every element has:
  "title"     – full paper title
  "authors"   – list of author strings
  "published" – date (YYYY-MM-DD or YYYY-MM)
  "venue"     – source (arXiv / conference / journal)
  "abstract"  – 2–3 sentence summary of the contribution
  "url"       – direct link to the paper

Return ONLY the JSON array.\
"""


# ─── JSON extraction ──────────────────────────────────────────────────────────

def _extract_json(response: anthropic.types.Message, label: str) -> list[dict]:
    """Pull the first text block out of a Claude response and parse it as JSON."""
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
        return []


# ─── API calls ────────────────────────────────────────────────────────────────

def fetch_foundational(topic: str, n: int, client: anthropic.Anthropic) -> list[FoundationalPaper]:
    """Call Claude with adaptive thinking to get foundational papers."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=FOUNDATIONAL_SYSTEM,
        messages=[{"role": "user", "content": FOUNDATIONAL_PROMPT.format(topic=topic, n=n)}],
    )
    return _validate(_extract_json(response, "foundational"), FoundationalPaper)


def fetch_recent(topic: str, n: int, days: int, client: anthropic.Anthropic) -> list[RecentPaper]:
    """Call Claude + web_search to find newly published papers."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=RECENT_SYSTEM,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        messages=[{"role": "user", "content": RECENT_PROMPT.format(topic=topic, n=n, days=days)}],
    )
    return _validate(_extract_json(response, "recent"), RecentPaper)


# ─── Orchestration (cache + concurrency) ──────────────────────────────────────

def get_foundational_papers(
    topic: str, n: int, client: anthropic.Anthropic, no_cache: bool
) -> list[FoundationalPaper]:
    key = _cache_key("foundational", topic, str(n))
    if not no_cache:
        hit = cache_load(key, FOUNDATIONAL_TTL_DAYS)
        if hit is not None:
            return _validate(hit, FoundationalPaper)
    papers = fetch_foundational(topic, n, client)
    cache_save(key, [p.model_dump() for p in papers])
    return papers


def get_recent_papers(
    topic: str, n: int, days: int, client: anthropic.Anthropic, no_cache: bool
) -> list[RecentPaper]:
    key = _cache_key("recent", topic, str(n), str(days))
    if not no_cache:
        hit = cache_load(key, RECENT_TTL_DAYS)
        if hit is not None:
            return _validate(hit, RecentPaper)
    papers = fetch_recent(topic, n, days, client)
    cache_save(key, [p.model_dump() for p in papers])
    return papers


# ─── Formatting ───────────────────────────────────────────────────────────────

DIVIDER  = "─" * 72
HDIVIDER = "═" * 72


def _wrap(text: str, indent: int = 4, width: int = 72) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def _print_header(title: str) -> None:
    print(f"\n{HDIVIDER}")
    print(f"  {title}")
    print(HDIVIDER)


def print_foundational(papers: list[FoundationalPaper]) -> None:
    _print_header("FOUNDATIONAL PAPERS")
    if not papers:
        print("  (no results returned)")
        return
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors)
        if len(authors) > 80:
            authors = authors[:77] + "…"
        print(f"\n  [{i}] {p.title}")
        print(f"      {authors} ({p.year})")
        if p.venue:
            print(f"      {p.venue}")
        if p.why:
            print(_wrap(p.why))
        if p.url:
            print(f"    → {p.url}")
        print(f"  {DIVIDER}")


def print_recent(papers: list[RecentPaper]) -> None:
    _print_header("RECENTLY PUBLISHED PAPERS")
    if not papers:
        print("  No recent papers found.")
        return
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors)
        if len(authors) > 80:
            authors = authors[:77] + "…"
        print(f"\n  [{i}] {p.title}")
        print(f"      {authors}")
        print(f"      Published: {p.published}  |  {p.venue}")
        if p.abstract:
            print(_wrap(p.abstract))
        if p.url:
            print(f"    → {p.url}")
        print(f"  {DIVIDER}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _cache_status(key: str, ttl_days: int) -> str:
    hit = cache_load(key, ttl_days)
    return "cache hit" if hit is not None else "fetching…"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper recommendations for a research topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python paper_recommender.py "transformer attention mechanisms"
              python paper_recommender.py "graph neural networks" -f 10 -r 15
              python paper_recommender.py "diffusion models" --days 180
              python paper_recommender.py "RL" --no-cache
        """),
    )
    parser.add_argument("topic", help="Research topic to search for")
    parser.add_argument("--foundational", "-f", type=int, default=8, metavar="N",
                        help="Number of foundational papers (default: 8)")
    parser.add_argument("--recent", "-r", type=int, default=10, metavar="N",
                        help="Number of recent papers (default: 10)")
    parser.add_argument("--days", "-d", type=int, default=365, metavar="DAYS",
                        help="Days back to search for recent papers (default: 365)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache and always fetch fresh results")
    args = parser.parse_args()

    topic = args.topic.strip()
    print(f"\nSearching paper recommendations for: '{topic}'\n")

    # Show cache status before any work starts
    f_key = _cache_key("foundational", topic, str(args.foundational))
    r_key = _cache_key("recent", topic, str(args.recent), str(args.days))
    f_status = "skip cache" if args.no_cache else _cache_status(f_key, FOUNDATIONAL_TTL_DAYS)
    r_status = "skip cache" if args.no_cache else _cache_status(r_key, RECENT_TTL_DAYS)
    print(f"  foundational papers  [{f_status}]")
    print(f"  recent papers        [{r_status}]")
    print()

    client = anthropic.Anthropic()

    # Run both queries concurrently (cache hits return immediately, API calls overlap)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_future = pool.submit(get_foundational_papers, topic, args.foundational, client, args.no_cache)
        r_future = pool.submit(get_recent_papers, topic, args.recent, args.days, client, args.no_cache)
        foundational = f_future.result()
        recent       = r_future.result()

    print_foundational(foundational)
    print_recent(recent)
    print()


if __name__ == "__main__":
    main()
