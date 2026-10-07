import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://api.github.com/graphql"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
PARTS = OUT / "parts"
PROJECTS = {
    "vscode": "microsoft/vscode",
    "react": "react/react",
    "kubernetes": "kubernetes/kubernetes",
    "django": "django/django",
    "numpy": "numpy/numpy",
    "rust": "rust-lang/rust",
}
YEARS = range(2019, 2025)
CEILING = 1000
BUDGET = float(os.environ.get("MAX_SECONDS", "0"))
WORKERS = int(os.environ.get("WORKERS", "6"))
PAGE = 50
T0 = time.time()
QUERY = """
query($q: String!, $n: Int!, $cursor: String) {
  search(query: $q, type: ISSUE, first: $n, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        createdAt
        mergedAt
        additions
        deletions
        changedFiles
        baseRefName
        isCrossRepository
        author { __typename login }
        mergedBy { __typename login }
        commits { totalCount }
        reviews { totalCount }
        comments { totalCount }
        labels(first: 30) { nodes { name } }
      }
    }
  }
  rateLimit { remaining resetAt }
}
"""


def token():
    t = os.environ.get("GITHUB_TOKEN")
    if not t and Path(".github_token").exists():
        t = Path(".github_token").read_text().strip()
    if not t:
        sys.exit("Set GITHUB_TOKEN or put the token in a file named .github_token")
    return t


def post(variables, tok, attempt=0):
    body = json.dumps({"query": QUERY, "variables": variables}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"bearer {tok}",
        "Content-Type": "application/json",
        "User-Agent": "oss-dora-fuzzy-fetch",
    })
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit("GitHub rejected the token (401)")
        if attempt >= 10:
            raise
        if e.code in (502, 504):
            time.sleep(2)
            return post(dict(variables, n=max(variables["n"] // 2, 10)), tok, attempt + 1)
        wait = int(e.headers.get("Retry-After") or min(60 * (attempt + 1), 600))
        print(f"    HTTP {e.code}, waiting {wait}s", flush=True)
        time.sleep(wait)
        return post(variables, tok, attempt + 1)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        if attempt >= 8:
            raise
        time.sleep(min(15 * (attempt + 1), 120))
        return post(variables, tok, attempt + 1)
    if data.get("errors"):
        if attempt >= 8:
            sys.exit(json.dumps(data["errors"])[:500])
        variables = dict(variables, n=max(variables["n"] // 2, 10))
        time.sleep(min(5 * (attempt + 1), 60))
        return post(variables, tok, attempt + 1)
    rl = data["data"]["rateLimit"]
    if rl["remaining"] < 50:
        reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
        wait = max((reset - datetime.now(timezone.utc)).total_seconds(), 0) + 10
        print(f"    rate limit reached, waiting {int(wait)}s", flush=True)
        time.sleep(wait)
    return data["data"]["search"]


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def window(repo, start, end, tok, audit):
    q = f"repo:{repo} is:pr is:merged merged:{iso(start)}..{iso(end)}"
    first = post({"q": q, "n": PAGE, "cursor": None}, tok)
    total = first["issueCount"]
    if total > CEILING:
        if end - start <= timedelta(seconds=1):
            sys.exit(f"window cannot be split further: {q}")
        mid = (start + (end - start) / 2).replace(microsecond=0)
        return (window(repo, start, mid, tok, audit)
                + window(repo, mid + timedelta(seconds=1), end, tok, audit))
    nodes, page = list(first["nodes"]), first
    while page["pageInfo"]["hasNextPage"]:
        page = post({"q": q, "n": PAGE, "cursor": page["pageInfo"]["endCursor"]}, tok)
        nodes += page["nodes"]
    nodes = [n for n in nodes if n and n.get("number") is not None]
    audit.append({"start": iso(start), "end": iso(end), "reported": total, "retrieved": len(nodes)})
    return nodes


def record(n):
    author = n.get("author") or {}
    merger = n.get("mergedBy") or {}
    return {
        "type": "pr",
        "number": n["number"],
        "created_at": n["createdAt"],
        "merged_at": n["mergedAt"],
        "additions": n["additions"],
        "deletions": n["deletions"],
        "changed_files": n["changedFiles"],
        "commits": n["commits"]["totalCount"],
        "reviews": n["reviews"]["totalCount"],
        "comments": n["comments"]["totalCount"],
        "author": author.get("login"),
        "author_type": author.get("__typename"),
        "merged_by": merger.get("login"),
        "merged_by_type": merger.get("__typename"),
        "base": n["baseRefName"],
        "from_fork": n["isCrossRepository"],
        "labels": [l["name"] for l in n["labels"]["nodes"]],
    }


def months():
    for y in YEARS:
        for m in range(1, 13):
            start = datetime(y, m, 1, tzinfo=timezone.utc)
            nxt = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
            yield f"{y}-{m:02d}", start, nxt - timedelta(seconds=1)


def fetch_month(pid, repo, label, start, end, tok):
    part = PARTS / f"{pid}_{label}.json"
    if part.exists() or (BUDGET and time.time() - T0 > BUDGET):
        return
    audit = []
    nodes = window(repo, start, end, tok, audit)
    seen, rows = set(), []
    for n in nodes:
        if n["number"] not in seen:
            seen.add(n["number"])
            rows.append(record(n))
    reported = sum(a["reported"] for a in audit)
    tmp = part.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"project": pid, "month": label, "reported": reported,
                               "retrieved": len(rows), "windows": audit, "prs": rows}))
    tmp.rename(part)
    flag = "" if reported == len(rows) else "  <-- mismatch"
    print(f"{pid} {label}: {len(rows)} of {reported}{flag}", flush=True)


def main():
    tok = token()
    PARTS.mkdir(parents=True, exist_ok=True)
    tasks = [(pid, repo, label, start, end) for label, start, end in months() for pid, repo in PROJECTS.items()]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda t: fetch_month(*t, tok), tasks))
    done = len(list(PARTS.glob("*.json")))
    if done < len(tasks):
        print(f"paused: {done} of {len(tasks)} project-months done; run again to continue")
        return
    summary = []
    for pid, repo in PROJECTS.items():
        seen = set()
        with open(OUT / f"{pid}.jsonl", "w") as f:
            f.write(json.dumps({"_meta": {"project_id": pid, "repo": repo, "source": "graphql-search",
                                          "fetched_at": iso(datetime.now(timezone.utc))}}) + "\n")
            for label, _, _ in months():
                p = json.loads((PARTS / f"{pid}_{label}.json").read_text())
                summary.append(f"{pid},{label},{p['reported']},{p['retrieved']},{len(p['windows'])}")
                for row in p["prs"]:
                    if row["number"] not in seen:
                        seen.add(row["number"])
                        f.write(json.dumps(row) + "\n")
    (OUT / "completeness.csv").write_text("project,month,reported,retrieved,windows\n" + "\n".join(summary) + "\n")
    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
