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
      foundational papers: 30-day TTL (stable)
      recent papers:        1-day TTL (use --no-cache to force refresh)
  - Unpaywall lookup: finds free legal copies for any paper with a DOI
  - Institutional proxy: --proxy prepends your EZproxy URL to DOI links
"""

import re
import sys
import json
import hashlib
import argparse
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
import anthropic
from pydantic import BaseModel, ValidationError


# ─── Cache ────────────────────────────────────────────────────────────────────

CACHE_DIR             = Path.home() / ".paper_recommender_cache"
FOUNDATIONAL_TTL_DAYS = 30
RECENT_TTL_DAYS       = 1


def _cache_key(*parts: str) -> str:
    raw = "|".join(str(p).lower().strip() for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_load(key: str, ttl_days: int) -> list[dict] | None:
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
        return None


def cache_save(key: str, papers: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "papers": papers,
    }, indent=2))


# ─── DOI / access helpers ─────────────────────────────────────────────────────

_DOI_RE = re.compile(r'10\.\d{4,}/[^\s"\'>,]+')


def extract_doi(url: str) -> str | None:
    """Return the bare DOI from a URL like https://doi.org/10.xxxx/yyy, or None."""
    m = _DOI_RE.search(url)
    return m.group(0).rstrip(".,)") if m else None


def unpaywall_lookup(doi: str, email: str) -> dict | None:
    """
    Query the Unpaywall API for open-access versions of a paper.
    Returns the Unpaywall response dict, or None if unavailable/network error.
    Unpaywall is free and legal — it only indexes openly available copies.
    """
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=6,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def best_oa_url(unpaywall: dict) -> str | None:
    """Extract the best open-access PDF or landing-page URL from Unpaywall data."""
    loc = unpaywall.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url") or None


def proxy_url(doi_url: str, proxy: str) -> str:
    """Prepend an EZproxy base URL to a DOI link.
    The proxy string is expected to end with '=' or '/' (e.g. '...login?url=').
    """
    return proxy + doi_url


# ─── Pydantic models ──────────────────────────────────────────────────────────

class FoundationalPaper(BaseModel):
    title:   str
    authors: list[str]
    year:    int
    venue:   str = ""
    why:     str = ""
    url:     str = ""


class RecentPaper(BaseModel):
    title:     str
    authors:   list[str]
    published: str
    venue:     str = ""
    abstract:  str = ""
    url:       str = ""


def _validate(raw: list[dict], model: type) -> list:
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
  "url"     – DOI link as https://doi.org/10.XXXX/YYYY (required if the paper
               has a DOI; fall back to arXiv abs URL only if no DOI exists)

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
  "url"       – DOI link (https://doi.org/10.XXXX/YYYY) if available,
                otherwise the arXiv abs URL

Return ONLY the JSON array.\
"""


# ─── JSON extraction ──────────────────────────────────────────────────────────

def _extract_json(response: anthropic.types.Message, label: str) -> list[dict]:
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text.strip()
            break

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


# ─── Claude API calls ─────────────────────────────────────────────────────────

def fetch_foundational(topic: str, n: int, client: anthropic.Anthropic) -> list[FoundationalPaper]:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=FOUNDATIONAL_SYSTEM,
        messages=[{"role": "user", "content": FOUNDATIONAL_PROMPT.format(topic=topic, n=n)}],
    )
    return _validate(_extract_json(response, "foundational"), FoundationalPaper)


def fetch_recent(topic: str, n: int, days: int, client: anthropic.Anthropic) -> list[RecentPaper]:
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


# ─── Access enrichment (Unpaywall + proxy) ────────────────────────────────────

def enrich_access(url: str, upw_email: str | None, proxy: str | None) -> list[str]:
    """
    Given a paper URL, return a list of access lines to print:
      - [open access] line if Unpaywall finds a free copy
      - [proxy] line if --proxy is set and the URL contains a DOI
    Returns an empty list when neither flag is active.
    """
    lines = []
    doi = extract_doi(url) if url else None

    # Unpaywall: find free legal copy
    if doi and upw_email:
        data = unpaywall_lookup(doi, upw_email)
        if data:
            oa_url = best_oa_url(data)
            if oa_url:
                lines.append(f"    [open access] {oa_url}")
            elif data.get("is_oa"):
                lines.append("    [open access] (free version available — check Unpaywall)")

    # Institutional proxy: wrap DOI link through EZproxy
    if doi and proxy:
        doi_link = f"https://doi.org/{doi}"
        lines.append(f"    [proxy access] {proxy_url(doi_link, proxy)}")

    return lines


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


def _print_access(url: str, upw_email: str | None, proxy: str | None) -> None:
    if url:
        print(f"    → {url}")
    for line in enrich_access(url, upw_email, proxy):
        print(line)


def print_foundational(
    papers: list[FoundationalPaper],
    upw_email: str | None,
    proxy: str | None,
) -> None:
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
        _print_access(p.url, upw_email, proxy)
        print(f"  {DIVIDER}")


def print_recent(
    papers: list[RecentPaper],
    upw_email: str | None,
    proxy: str | None,
) -> None:
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
        _print_access(p.url, upw_email, proxy)
        print(f"  {DIVIDER}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _cache_status(key: str, ttl_days: int) -> str:
    return "cache hit" if cache_load(key, ttl_days) is not None else "fetching…"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper recommendations for a research topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python paper_recommender.py "transformer attention mechanisms"
              python paper_recommender.py "graph neural networks" -f 10 -r 15
              python paper_recommender.py "diffusion models" --days 180 --no-cache

              # Find free legal copies via Unpaywall:
              python paper_recommender.py "RL" --email you@university.edu

              # Route DOI links through your institution's proxy:
              python paper_recommender.py "RL" --proxy "https://proxy.myuniversity.edu/login?url="

              # Both at once:
              python paper_recommender.py "RL" \\
                --email you@university.edu \\
                --proxy "https://proxy.myuniversity.edu/login?url="
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
    parser.add_argument(
        "--email", metavar="EMAIL",
        help="Your email address — enables Unpaywall lookup for free legal copies of papers",
    )
    parser.add_argument(
        "--proxy", metavar="URL",
        help=(
            "Your institution's EZproxy base URL. DOI links will be rewritten "
            "so they route through your library's subscription access. "
            "Example: https://proxy.myuniversity.edu/login?url="
        ),
    )
    args = parser.parse_args()

    topic = args.topic.strip()
    print(f"\nSearching paper recommendations for: '{topic}'\n")

    if args.email:
        print(f"  Unpaywall lookup enabled  ({args.email})")
    if args.proxy:
        print(f"  Institutional proxy       {args.proxy}")
    if args.email or args.proxy:
        print()

    f_key = _cache_key("foundational", topic, str(args.foundational))
    r_key = _cache_key("recent", topic, str(args.recent), str(args.days))
    f_status = "skip cache" if args.no_cache else _cache_status(f_key, FOUNDATIONAL_TTL_DAYS)
    r_status = "skip cache" if args.no_cache else _cache_status(r_key, RECENT_TTL_DAYS)
    print(f"  foundational papers  [{f_status}]")
    print(f"  recent papers        [{r_status}]")
    print()

    client = anthropic.Anthropic()

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_future = pool.submit(get_foundational_papers, topic, args.foundational, client, args.no_cache)
        r_future = pool.submit(get_recent_papers, topic, args.recent, args.days, client, args.no_cache)
        foundational = f_future.result()
        recent       = r_future.result()

    print_foundational(foundational, args.email, args.proxy)
    print_recent(recent, args.email, args.proxy)
    print()


if __name__ == "__main__":
    main()
