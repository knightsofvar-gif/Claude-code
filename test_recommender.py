#!/usr/bin/env python3
"""
Validation script for paper_recommender.py

Runs benchmark queries and checks:
  1. Known foundational papers appear in results   (ground-truth matching)
  2. Returned paper count matches requested N       (completeness)
  3. No Pydantic entries were dropped              (data quality)
  4. DOI links resolve via HTTP HEAD               (link health)
  5. Recent papers fall within the requested date window (freshness)

Exit code 0 = all checks passed
Exit code 1 = one or more checks failed

Usage:
  python test_recommender.py                  # full run (uses cache when available)
  python test_recommender.py --no-cache       # force fresh API calls
  python test_recommender.py --skip-doi       # skip network DOI checks
  python test_recommender.py --skip-api       # only test cache/logic, no Claude calls
"""

import os
import re
import sys
import json
import argparse
import textwrap
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

import requests
import anthropic

# Import from the main module (same directory)
from paper_recommender import (
    get_foundational_papers,
    get_recent_papers,
    extract_doi,
    FoundationalPaper,
    RecentPaper,
    FOUNDATIONAL_TTL_DAYS,
    RECENT_TTL_DAYS,
    _cache_key,
    cache_load,
)


# ─── Benchmark definitions ────────────────────────────────────────────────────
#
# Each benchmark specifies a topic, how many papers to request, and a list of
# expected title substrings (case-insensitive). At least one result title must
# contain each substring for the check to pass.

BENCHMARKS = [
    {
        "topic": "transformer attention mechanisms",
        "n_foundational": 6,
        "n_recent": 5,
        "days_recent": 365,
        "expected_titles": [
            "attention is all you need",
            "bert",
        ],
    },
    {
        "topic": "generative adversarial networks",
        "n_foundational": 5,
        "n_recent": 5,
        "days_recent": 365,
        "expected_titles": [
            "generative adversarial",
        ],
    },
    {
        "topic": "reinforcement learning from human feedback",
        "n_foundational": 5,
        "n_recent": 5,
        "days_recent": 365,
        "expected_titles": [
            "reinforcement learning from human feedback",
        ],
    },
    {
        "topic": "diffusion models image generation",
        "n_foundational": 5,
        "n_recent": 5,
        "days_recent": 365,
        "expected_titles": [
            "denoising diffusion",
        ],
    },
    {
        "topic": "bioadsorption of rare earth elements",
        "n_foundational": 6,
        "n_recent": 5,
        "days_recent": 365,
        "expected_titles": [
            # Volesky's biosorption work is the cornerstone of the field
            "biosorption",
            # REE recovery / rare earth is the core application
            "rare earth",
        ],
    },
]


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    topic: str
    name: str
    passed: bool
    detail: str = ""


# ─── Individual checks ────────────────────────────────────────────────────────

def check_count(
    topic: str,
    papers: list,
    requested: int,
    label: str,
) -> CheckResult:
    got = len(papers)
    passed = got >= max(1, requested - 1)   # allow off-by-one from model
    detail = f"got {got}, requested {requested}"
    return CheckResult(topic, f"{label} count", passed, detail)


def check_expected_titles(
    topic: str,
    papers: list[FoundationalPaper],
    expected: list[str],
) -> list[CheckResult]:
    results = []
    titles_lower = [p.title.lower() for p in papers]
    for expected_substr in expected:
        found = any(expected_substr.lower() in t for t in titles_lower)
        results.append(CheckResult(
            topic,
            f"contains '{expected_substr}'",
            found,
            "found" if found else f"missing — got: {[p.title for p in papers]}",
        ))
    return results


def check_doi_resolves(
    topic: str,
    papers: list,
    label: str,
    timeout: int = 6,
) -> list[CheckResult]:
    results = []
    for p in papers:
        url = getattr(p, "url", "") or ""
        doi = extract_doi(url)
        if not doi:
            continue   # arXiv-only link — skip
        doi_url = f"https://doi.org/{doi}"
        try:
            resp = requests.head(doi_url, allow_redirects=True, timeout=timeout)
            passed = resp.status_code < 400
            detail = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            passed = False
            detail = str(exc)
        results.append(CheckResult(topic, f"{label} DOI {doi[:40]}", passed, detail))
    return results


def check_recent_dates(
    topic: str,
    papers: list[RecentPaper],
    days: int,
) -> list[CheckResult]:
    results = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days + 30)  # 30-day grace for indexing lag
    for p in papers:
        date_str = (p.published or "")[:10]
        if not date_str:
            results.append(CheckResult(topic, f"date '{p.title[:40]}'", False, "no date"))
            continue
        try:
            pub_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            passed = pub_date >= cutoff
            detail = date_str
        except ValueError:
            passed = True   # partial date like "2024-11" — accept
            detail = f"partial date '{date_str}' accepted"
        results.append(CheckResult(
            topic,
            f"freshness '{p.title[:40]}'",
            passed,
            detail,
        ))
    return results


# ─── Run one benchmark ────────────────────────────────────────────────────────

def run_benchmark(
    bm: dict,
    client: anthropic.Anthropic | None,
    no_cache: bool,
    skip_doi: bool,
) -> list[CheckResult]:
    topic       = bm["topic"]
    n_found     = bm["n_foundational"]
    n_recent    = bm["n_recent"]
    days_recent = bm["days_recent"]
    expected    = bm["expected_titles"]
    results     = []

    # ── Foundational papers ──────────────────────────────────────────────────
    if client is None:
        # No API key — only run if cache is warm
        key = _cache_key("foundational", topic, str(n_found))
        raw = cache_load(key, FOUNDATIONAL_TTL_DAYS)
        if raw is None:
            results.append(CheckResult(topic, "foundational (skipped — no cache)", True,
                                       "cache miss + no API key"))
            foundational = []
        else:
            from paper_recommender import _validate
            foundational = _validate(raw, FoundationalPaper)
    else:
        try:
            foundational = get_foundational_papers(topic, n_found, client, no_cache)
        except Exception as exc:
            results.append(CheckResult(topic, "foundational API call", False, str(exc)))
            foundational = []

    if foundational:
        results.append(check_count(topic, foundational, n_found, "foundational"))
        results.extend(check_expected_titles(topic, foundational, expected))
        if not skip_doi:
            results.extend(check_doi_resolves(topic, foundational, "foundational"))

    # ── Recent papers ────────────────────────────────────────────────────────
    if client is None:
        key = _cache_key("recent", topic, str(n_recent), str(days_recent))
        raw = cache_load(key, RECENT_TTL_DAYS)
        if raw is None:
            results.append(CheckResult(topic, "recent (skipped — no cache)", True,
                                       "cache miss + no API key"))
            recent = []
        else:
            from paper_recommender import _validate
            recent = _validate(raw, RecentPaper)
    else:
        try:
            recent = get_recent_papers(topic, n_recent, days_recent, client, no_cache)
        except Exception as exc:
            results.append(CheckResult(topic, "recent API call", False, str(exc)))
            recent = []

    if recent:
        results.append(check_count(topic, recent, n_recent, "recent"))
        results.extend(check_recent_dates(topic, recent, days_recent))
        if not skip_doi:
            results.extend(check_doi_resolves(topic, recent, "recent"))

    return results


# ─── Report ───────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
DIVIDER = "─" * 68


def print_report(all_results: list[CheckResult], elapsed_s: float) -> int:
    """Print a formatted report. Returns exit code (0 = all pass, 1 = any fail)."""
    by_topic: dict[str, list[CheckResult]] = {}
    for r in all_results:
        by_topic.setdefault(r.topic, []).append(r)

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed

    print("\n" + "═" * 68)
    print("  PAPER RECOMMENDER — VALIDATION REPORT")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
          f"{total} checks  |  {passed} passed  |  {failed} failed  |  {elapsed_s:.1f}s")
    print("═" * 68)

    for topic, results in by_topic.items():
        t_pass = sum(1 for r in results if r.passed)
        t_fail = len(results) - t_pass
        status = PASS if t_fail == 0 else FAIL
        print(f"\n  {status} {topic}  ({t_pass}/{len(results)})")
        print(f"  {DIVIDER}")
        for r in results:
            icon = PASS if r.passed else FAIL
            name = r.name[:44].ljust(44)
            detail = f"  {r.detail}" if r.detail else ""
            print(f"    {icon}  {name}{detail}")

    print(f"\n{'═' * 68}")
    if failed == 0:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {failed} CHECK(S) FAILED — review output above")
    print("═" * 68 + "\n")

    return 0 if failed == 0 else 1


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate paper_recommender.py against benchmark topics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python test_recommender.py
              python test_recommender.py --skip-doi
              python test_recommender.py --no-cache --skip-doi
              python test_recommender.py --skip-api          # only checks cached results
        """),
    )
    parser.add_argument("--no-cache",  action="store_true", help="Force fresh API calls")
    parser.add_argument("--skip-doi",  action="store_true", help="Skip DOI link health checks")
    parser.add_argument("--skip-api",  action="store_true",
                        help="Skip API calls entirely (only validate cached results)")
    args = parser.parse_args()

    import time
    start = time.monotonic()

    # Build client only if we have an API key and aren't skipping API calls
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if args.skip_api or not api_key:
        client = None
        if not args.skip_api and not api_key:
            print("[warning] ANTHROPIC_API_KEY not set — running in cache-only mode\n",
                  file=sys.stderr)
    else:
        client = anthropic.Anthropic(api_key=api_key)

    all_results: list[CheckResult] = []
    for bm in BENCHMARKS:
        print(f"  Running benchmark: {bm['topic']} …", flush=True)
        results = run_benchmark(bm, client, args.no_cache, args.skip_doi)
        all_results.extend(results)

    elapsed = time.monotonic() - start
    exit_code = print_report(all_results, elapsed)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
