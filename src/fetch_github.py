"""
GitHub fetcher for the OSS longitudinal study.
==============================================

Fetches the raw signals needed by `compute_metrics.py` and writes them to
`data/raw/{project_id}.jsonl`. Reviewers run this with their own GitHub
token to verify the bundled sample against live data.

What gets fetched per project:
  - All merged pull requests with timestamps, additions, deletions, commits.
  - All releases (tags) within the study window.

Rate limiting: PRs are fetched via the Search API (30 req/min) with one
query per quarter, so GitHub does the date filtering server-side. PR detail
calls (additions/deletions) use the core API (5 000 req/hr). Both limits
are handled automatically.

Usage:
    export GITHUB_TOKEN=ghp_xxx        # personal access token, public repo scope
    python src/fetch_github.py         # fetch all projects in config/projects.yaml
    python src/fetch_github.py vscode  # fetch one project
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CONFIG = ROOT / "config" / "projects.yaml"

GITHUB_API  = "https://api.github.com"
SEARCH_API  = f"{GITHUB_API}/search/issues"
PER_PAGE    = 100
MAX_SEARCH_PAGES   = 10   # Search API hard cap: 1,000 results per query
MAX_PAGES_RELEASES = 10
DETAIL_WORKERS     = 5    # parallel threads for PR detail fetches

_rate_lock = threading.Lock()  # ensures only one thread sleeps/checks at a time


def _headers() -> dict:
    """Build auth headers. Token is required to avoid 60-req/hour limit."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit(
            "ERROR: GITHUB_TOKEN env var is required.\n"
            "Create a token at https://github.com/settings/tokens (no scopes "
            "needed for public repos) and export it before running."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET with thread-safe rate-limit awareness.
    Threshold is 10% of the limit so it works for both the core API (5 000/hr)
    and the search API (30/min). The lock prevents multiple threads from all
    deciding simultaneously that remaining is fine and bursting past the limit."""
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    with _rate_lock:
        limit     = int(r.headers.get("X-RateLimit-Limit",     "5000"))
        remaining = int(r.headers.get("X-RateLimit-Remaining", "1000"))
        if remaining < max(10, limit // 10):
            reset = int(r.headers.get("X-RateLimit-Reset", str(int(time.time()) + 60)))
            delay = max(2, reset - int(time.time()))
            print(f"  [rate-limit] {remaining}/{limit} remaining, sleeping {delay}s", flush=True)
            time.sleep(delay)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API {r.status_code}: {r.text[:200]}")
    return r


def _fetch_detail(pr_url: str) -> dict:
    """Fetch additions/deletions/commits for a single PR (used by thread pool)."""
    detail = _get(pr_url).json()
    return {
        "number":          detail["number"],
        "created_at":      detail["created_at"],
        "merged_at":       detail["merged_at"],
        "additions":       detail.get("additions", 0),
        "deletions":       detail.get("deletions", 0),
        "commits":         detail.get("commits", 0),
        "review_comments": detail.get("review_comments", 0),
    }


def _quarter_windows(since: str, until: str):
    """Yield (start_str, end_str) for each calendar quarter that overlaps [since, until]."""
    start = date.fromisoformat(since)
    end   = date.fromisoformat(until)
    y, m  = start.year, ((start.month - 1) // 3) * 3 + 1
    while True:
        q_start = date(y, m, 1)
        next_m  = m + 3
        q_end   = date(y + next_m // 12, next_m % 12 or 12, 1) - timedelta(days=1)
        window_start = max(q_start, start)
        window_end   = min(q_end,   end)
        if window_start > end:
            break
        yield window_start.isoformat(), window_end.isoformat()
        m += 3
        if m > 12:
            m, y = m - 12, y + 1


def fetch_pulls(owner: str, repo: str, since: str, until: str, out_file) -> int:
    """Fetch merged PRs in [since, until] via the Search API (one query per quarter).

    Phase 1 (sequential): one Search API call per quarter to collect PR URLs.
    Phase 2 (parallel):   DETAIL_WORKERS threads fetch additions/deletions concurrently.
                          Each completed PR is written and flushed to out_file immediately
                          so progress is not lost on crash.

    Returns the number of PRs written.
    """
    quarters = list(_quarter_windows(since, until))
    write_lock = threading.Lock()

    # --- Phase 1: collect all PR detail URLs via Search API (sequential) --------
    pr_urls: list[tuple[int, str]] = []  # (pr_number, detail_url)
    for q_idx, (q_start, q_end) in enumerate(quarters, 1):
        print(f"  [{q_idx}/{len(quarters)}] {q_start}..{q_end} — searching...", flush=True)
        query = f"repo:{owner}/{repo} type:pr is:merged merged:{q_start}..{q_end}"
        page  = 1
        while page <= MAX_SEARCH_PAGES:
            r     = _get(SEARCH_API, params={"q": query, "per_page": PER_PAGE, "page": page})
            data  = r.json()
            total = data.get("total_count", 0)
            if page == 1:
                print(f"         {total} PRs found"
                      + (" — GitHub caps at 1 000, first 1 000 will be used" if total > 1000 else ""),
                      flush=True)
            items = data.get("items", [])
            for item in items:
                pr_urls.append((item["number"], item["pull_request"]["url"]))
            if len(items) < PER_PAGE:
                break
            page += 1

    # --- Phase 2: fetch PR details in parallel, write each one immediately ------
    total_prs = len(pr_urls)
    print(f"  fetching details for {total_prs} PRs ({DETAIL_WORKERS} threads)...", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        futures = {pool.submit(_fetch_detail, url): num for num, url in pr_urls}
        for future in as_completed(futures):
            pr    = future.result()
            pr_num = futures[future]
            with write_lock:
                out_file.write(json.dumps({"type": "pr", **pr}) + "\n")
                out_file.flush()
            done += 1
            print(f"  {done}/{total_prs} details fetched (last: PR #{pr_num})   ",
                  end="\r", flush=True)
    print(f"  {total_prs} PRs fetched.{' ' * 40}", flush=True)
    return total_prs


def fetch_releases(owner: str, repo: str, since: str, until: str) -> list[dict]:
    """Fetch releases whose published date falls in [since, until]."""
    out: list[dict] = []
    since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    until_dt = datetime.fromisoformat(until).replace(tzinfo=timezone.utc)
    url = f"{GITHUB_API}/repos/{owner}/{repo}/releases"
    page = 1
    while page <= MAX_PAGES_RELEASES:
        r = _get(url, params={"per_page": PER_PAGE, "page": page})
        batch = r.json()
        if not batch:
            break
        for rel in batch:
            pub = rel.get("published_at")
            if not pub:
                continue
            pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            if since_dt <= pub_dt <= until_dt:
                out.append({
                    "tag_name":     rel["tag_name"],
                    "published_at": pub,
                    "prerelease":   rel.get("prerelease", False),
                })
        page += 1
    return out


def fetch_project(project: dict, since: str, until: str) -> None:
    """Fetch one project and write a single jsonl file."""
    pid    = project["id"]
    owner  = project["owner"]
    repo   = project["repo"]
    out_path = RAW_DIR / f"{pid}.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {pid} ({owner}/{repo}) ===")

    # Open the file immediately so each PR is persisted as it arrives.
    # A crash mid-fetch loses only the in-flight batch, not all prior work.
    with open(out_path, "w") as f:
        f.write(json.dumps({"_meta": {"project_id": pid, "owner": owner, "repo": repo,
                                       "fetched_at": datetime.now(timezone.utc).isoformat()}}) + "\n")
        f.flush()

        print(f"  fetching merged PRs...")
        n_prs = fetch_pulls(owner, repo, since, until, out_file=f)
        print(f"  {n_prs} PRs in window")

        print(f"  fetching releases...")
        rels = fetch_releases(owner, repo, since, until)
        for rel in rels:
            f.write(json.dumps({"type": "release", **rel}) + "\n")
        print(f"  {len(rels)} releases in window")

    print(f"  wrote {out_path}")


def main() -> int:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    since = cfg["study_window"]["start"]
    until = cfg["study_window"]["end"]

    only = sys.argv[1] if len(sys.argv) > 1 else None

    for project in cfg["projects"]:
        if only and project["id"] != only:
            continue
        try:
            fetch_project(project, since, until)
        except Exception as e:
            print(f"  ERROR for {project['id']}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
