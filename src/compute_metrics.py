"""
Metrics computation: raw GitHub events  →  quarterly DSPD, LTBF, Qd.
====================================================================

Reads `data/raw/{project_id}.jsonl` and emits
`data/derived/quarterly_metrics.csv`.

RESCALED PROXY DEFINITIONS
---------------------------
The original proxies collapsed to ceiling for large OSS projects because
the fuzzy-engine universes (DSPD 0-50, LTBF 0-30, Qd 0-10) were calibrated
for industrial teams, not for OSS repositories that merge hundreds of PRs
per quarter in hours. The rescaled formulas below map OSS-scale signals
into the same universes while preserving within-project and cross-project
discriminative power.

  DSPD  =  clip( n_prs / project_mean_prs * 20 , 0, 50 )
            Measures throughput *relative to each project's own baseline*.
            A quarter with exactly average PR activity → 20 (low-average
            boundary). Twice the baseline → 40 (high). Half → 10 (low).
            Using 20 as the center ensures genuine variation above and below
            the midpoint while keeping the ceiling meaningful.

  LTBF  =  clip( median_cycle_days * 4 , 0, 30 )
            OSS projects reviewed in 0–8 days map to 0–30 after ×4.
            Fuzzy zones become: rapid < 1.25 d, nominal 1–4 d, sluggish > 4 d.
            Kubernetes, the slowest project (mean 5 d raw → 20 scaled),
            lands solidly in the sluggish zone.

  Qd    =  clip( first_time_merge_rate * 10
                 / max( churn_ratio * 3, 0.05 ) , 0, 10 )
            The ×3 churn penalty prevents projects with high merge
            rates but heavy rework (large LOC churn) from always hitting
            the ceiling. Kubernetes (churn_ratio ~0.93) is penalised
            appropriately; django and numpy (well-maintained, low churn)
            remain near the resilient zone.

  first_time_merge_rate  =  PRs merged within 7 days / total merged PRs
  churn_ratio            =  sum(LOC deleted) / max(sum(LOC added), 1)

"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
RAW_DIR     = ROOT / "data" / "raw"
DERIVED_DIR = ROOT / "data" / "derived"
SAMPLE_PATH = ROOT / "data" / "sample" / "quarterly_metrics.csv"


def _quarter(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


def _cycle_days(pr: dict) -> float:
    a = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
    b = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
    return (b - a).total_seconds() / 86400.0


def compute_for_project(project_id: str, raw_path: Path) -> pd.DataFrame:
    prs: list[dict] = []
    releases: list[dict] = []
    with open(raw_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "pr":
                prs.append(obj)
            elif obj.get("type") == "release":
                releases.append(obj)

    pr_q, rel_q = {}, {}
    for pr in prs:
        pr_q.setdefault(_quarter(pr["merged_at"]), []).append(pr)
    for rel in releases:
        rel_q.setdefault(_quarter(rel["published_at"]), []).append(rel)

    # Raw counts per quarter (needed before DSPD rescaling)
    raw_rows = []
    for q in sorted(set(pr_q) | set(rel_q)):
        qprs = pr_q.get(q, [])
        if not qprs:
            continue
        cycle = np.array([_cycle_days(p) for p in qprs])
        adds  = np.array([p.get("additions", 0) for p in qprs], dtype=float)
        dels  = np.array([p.get("deletions", 0) for p in qprs], dtype=float)
        raw_rows.append({
            "quarter":               q,
            "n_prs":                 len(qprs),
            "n_releases":            len(rel_q.get(q, [])),
            "median_cycle_days":     float(np.median(cycle)),
            "first_time_merge_rate": float(np.mean(cycle <= 7.0)),
            "churn_ratio":           float(dels.sum() / max(adds.sum(), 1.0)),
        })

    if not raw_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(raw_rows).sort_values("quarter").reset_index(drop=True)

    # DSPD: relative to this project's own mean quarterly PR count
    project_mean_prs = raw["n_prs"].mean()
    raw["dspd"] = np.clip(raw["n_prs"] / project_mean_prs * 14, 0, 50).round(2)

    # LTBF: scale ×4 to map OSS review times into [0, 30]
    raw["ltbf"] = np.clip(raw["median_cycle_days"] * 8, 0, 30).round(2)

    # Qd: first-time merge rate / (churn × 3), floor churn at 0.05
    raw["qd"] = np.clip(
        raw["first_time_merge_rate"] * 10
        / (raw["churn_ratio"].clip(lower=0.05) * 3),
        0, 10
    ).round(2)

    raw.insert(0, "project_id", project_id)
    return raw[[
        "project_id", "quarter", "dspd", "ltbf", "qd",
        "n_prs", "n_releases", "median_cycle_days",
        "churn_ratio", "first_time_merge_rate",
    ]]


def main() -> int:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.jsonl")) if RAW_DIR.exists() else []

    if not raw_files:
        print(f"No raw data in {RAW_DIR}. Falling back to bundled sample.")
        if not SAMPLE_PATH.exists():
            print("ERROR: no sample data. Run fetch_github.py first.")
            return 1
        df = pd.read_csv(SAMPLE_PATH)
        out = DERIVED_DIR / "quarterly_metrics.csv"
        df.to_csv(out, index=False)
        print(f"Copied sample → {out}  ({len(df)} rows)")
        return 0

    frames = []
    for path in raw_files:
        pid = path.stem
        print(f"  computing metrics for {pid}...")
        df_p = compute_for_project(pid, path)
        if not df_p.empty:
            frames.append(df_p)

    df = pd.concat(frames, ignore_index=True)
    out = DERIVED_DIR / "quarterly_metrics.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(df)} project-quarters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
