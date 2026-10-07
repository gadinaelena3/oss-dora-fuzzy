import json
import os
import sys
import urllib.request
from pathlib import Path

REPOS = ["microsoft/vscode", "react/react", "kubernetes/kubernetes", "django/django", "numpy/numpy", "rust-lang/rust"]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/search_coverage.csv")


def gql(query, tok):
    req = urllib.request.Request("https://api.github.com/graphql", data=json.dumps({"query": query}).encode(),
                                 headers={"Authorization": f"bearer {tok}", "User-Agent": "oss-dora-fuzzy"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]


def main():
    tok = os.environ.get("GITHUB_TOKEN") or Path(".github_token").read_text().strip()
    rows = ["repository,merged_prs_repository,merged_prs_search,not_returned,share_pct"]
    for repo in REPOS:
        owner, name = repo.split("/")
        total = gql('{ repository(owner:"%s",name:"%s"){ pullRequests(states:MERGED){ totalCount } } }' % (owner, name), tok)["repository"]["pullRequests"]["totalCount"]
        found = gql('{ search(query:"repo:%s is:pr is:merged", type:ISSUE, first:1){ issueCount } }' % repo, tok)["search"]["issueCount"]
        rows.append(f"{repo},{total},{found},{total - found},{100 * (total - found) / total:.2f}")
    OUT.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
