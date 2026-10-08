import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
CONFIG = ROOT / "config" / "projects.yaml"
ANCHORS = {"dspd": 12.0, "ltbf": 7.0, "qd": 3.0}
BOT_LOGINS = {"rustbot"}
BOT_SUFFIXES = ("-bot", "-robot", "[bot]")
CHURN_FLOOR = 0.05
DEFAULT_BRANCHES = ("main", "master")


def is_bot(pr):
    login = (pr.get("author") or "").lower()
    return pr.get("author_type") == "Bot" or login in BOT_LOGINS or login.endswith(BOT_SUFFIXES)


def load_prs():
    rows = []
    for path in sorted(RAW.glob("*.jsonl")):
        for line in open(path):
            o = json.loads(line)
            if o.get("type") != "pr":
                continue
            rows.append({
                "project_id": path.stem,
                "created": o["created_at"],
                "merged": o["merged_at"],
                "additions": o["additions"],
                "deletions": o["deletions"],
                "bot": is_bot(o),
                "default_base": o.get("base") in DEFAULT_BRANCHES,
                "from_fork": bool(o.get("from_fork")),
                "self_merged": o.get("author") is not None and o.get("author") == o.get("merged_by"),
                "no_review": o.get("reviews", 0) == 0,
            })
    d = pd.DataFrame(rows)
    d["created"] = pd.to_datetime(d.created).dt.tz_localize(None)
    d["merged"] = pd.to_datetime(d.merged).dt.tz_localize(None)
    d["days"] = (d.merged - d.created).dt.total_seconds() / 86400
    d["quarter"] = d.merged.dt.to_period("Q").astype(str)
    return d


def signals(prs, include_bots=False):
    bots = prs[prs.bot].groupby(["project_id", "quarter"]).size().rename("n_bot_prs")
    d = prs if include_bots else prs[~prs.bot]
    g = d.groupby(["project_id", "quarter"])
    s = pd.DataFrame({
        "n_prs": g.size(),
        "median_cycle_days": g.days.median(),
        "first_time_merge_rate": g.days.apply(lambda x: (x <= 7).mean()),
        "churn_ratio": g.deletions.sum() / g.additions.sum().clip(lower=1),
        "share_under_1h": g.days.apply(lambda x: (x < 1 / 24).mean()),
        "share_from_fork": g.from_fork.mean(),
        "share_self_merged": g.self_merged.mean(),
        "share_no_review": g.no_review.mean(),
    }).join(bots).fillna({"n_bot_prs": 0}).reset_index()
    s["n_bot_prs"] = s.n_bot_prs.astype(int)
    return s


def pre_ai_quarters():
    cfg = yaml.safe_load(open(CONFIG))
    return set(cfg["eras"]["pre_ai"]["quarters"])


def calibrate(s, quarters=None):
    pre = s[s.quarter.isin(quarters or pre_ai_quarters())]
    base = pre.groupby("project_id").n_prs.mean()
    volume = pre.n_prs / pre.project_id.map(base)
    quality = pre.first_time_merge_rate / pre.churn_ratio.clip(lower=CHURN_FLOOR)
    k = {
        "k_dspd": round(ANCHORS["dspd"] / volume.median(), 2),
        "k_ltbf": round(ANCHORS["ltbf"] / pre.median_cycle_days.median(), 2),
        "k_qd": round(10 * quality.median() / ANCHORS["qd"], 2),
    }
    return k, base


def proxies(s, base, k_dspd, k_ltbf, k_qd):
    out = s.copy()
    out["pre_ai_mean_prs"] = out.project_id.map(base).round(2)
    out["dspd"] = np.clip(out.n_prs / out.project_id.map(base) * k_dspd, 0, 50).round(2)
    out["ltbf"] = np.clip(out.median_cycle_days * k_ltbf, 0, 30).round(2)
    out["qd"] = np.clip(out.first_time_merge_rate * 10 / (out.churn_ratio.clip(lower=CHURN_FLOOR) * k_qd), 0, 10).round(2)
    return out


def main():
    include_bots = "--include-bots" in sys.argv
    DERIVED.mkdir(parents=True, exist_ok=True)
    s = signals(load_prs(), include_bots)
    k, base = calibrate(s)
    out = proxies(s, base, **k)
    cols = ["project_id", "quarter", "dspd", "ltbf", "qd", "n_prs", "n_bot_prs", "pre_ai_mean_prs",
            "median_cycle_days", "first_time_merge_rate", "churn_ratio",
            "share_under_1h", "share_from_fork", "share_self_merged", "share_no_review"]
    out[cols].to_csv(DERIVED / "quarterly_metrics.csv", index=False)
    (DERIVED / "calibration.json").write_text(json.dumps({
        **k, "anchors": ANCHORS, "bots_included": include_bots,
        "pre_ai_mean_prs": {p: round(v, 2) for p, v in base.items()},
    }, indent=2))
    print(json.dumps(k), f"{len(out)} project-quarters, {int(out.n_prs.sum())} pull requests, {int(out.n_bot_prs.sum())} bot-authored")


if __name__ == "__main__":
    main()
