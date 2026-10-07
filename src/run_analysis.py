import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fuzzy_engine import infer

CONFIG = ROOT / "config" / "projects.yaml"
DERIVED = ROOT / "data" / "derived" / "quarterly_metrics.csv"
RESULTS = ROOT / "results"
ALPHA = 0.05 / 6


def load_eras():
    cfg = yaml.safe_load(open(CONFIG))
    return {era: meta["quarters"] for era, meta in cfg["eras"].items()}


def apply_dss(df):
    rows = []
    for _, r in df.iterrows():
        result = infer(r["dspd"], r["ltbf"], r["qd"])
        fired = [a["id"] for a in result.rule_activations if a["firing_strength"] > 0]
        rows.append({**r.to_dict(), "phs": result.phs, "linguistic_state": result.linguistic_state,
                     "rules_fired": ",".join(fired) if fired else "NONE", "n_rules_fired": len(fired)})
    return pd.DataFrame(rows)


def cliffs_delta(a, b):
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def era_comparison(df, eras):
    pre, ai = set(eras["pre_ai"]), set(eras["ai_era"])
    rows = []
    for project_id, sub in df.groupby("project_id"):
        a = sub.loc[sub["quarter"].isin(pre), "phs"].values
        b = sub.loc[sub["quarter"].isin(ai), "phs"].values
        u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
        rows.append({
            "project_id": project_id, "n_pre": len(a), "n_ai": len(b),
            "phs_pre_mean": round(float(np.mean(a)), 2), "phs_pre_median": round(float(np.median(a)), 2),
            "phs_ai_mean": round(float(np.mean(b)), 2), "phs_ai_median": round(float(np.median(b)), 2),
            "phs_shift": round(float(np.mean(b) - np.mean(a)), 2),
            "mwu_U": round(float(u), 2), "mwu_p": round(float(p), 4),
            "cliffs_delta": round(float(cliffs_delta(b, a)), 3),
            "significant_05": bool(p < 0.05), "significant_bonferroni": bool(p < ALPHA),
        })
    return pd.DataFrame(rows)


def discriminative_power(df):
    phs = df["phs"].values
    counts = Counter(df["linguistic_state"].values)
    probs = np.array(list(counts.values())) / len(df)
    entropy = float(-(probs * np.log2(probs)).sum())
    h, p = stats.kruskal(*[g["phs"].values for _, g in df.groupby("project_id")])
    return {
        "n_observations": len(df),
        "phs_mean": round(float(np.mean(phs)), 2), "phs_std": round(float(np.std(phs)), 2),
        "phs_variance": round(float(np.var(phs)), 2),
        "phs_iqr": round(float(np.percentile(phs, 75) - np.percentile(phs, 25)), 2),
        "phs_min": round(float(np.min(phs)), 2), "phs_max": round(float(np.max(phs)), 2),
        "state_distribution": dict(counts),
        "entropy_bits": round(entropy, 3), "entropy_normalised": round(entropy / float(np.log2(5)), 3),
        "kruskal_H": round(float(h), 2), "kruskal_p": float(p),
        "no_rule_fallback_pct": round(float((df["rules_fired"] == "NONE").mean() * 100), 2),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = apply_dss(pd.read_csv(DERIVED))
    df.to_csv(RESULTS / "phs_timeseries.csv", index=False)
    era = era_comparison(df, load_eras())
    era.to_csv(RESULTS / "era_comparison.csv", index=False)
    dp = discriminative_power(df)
    (RESULTS / "discriminative_power.json").write_text(json.dumps(dp, indent=2))
    print(json.dumps({k: v for k, v in dp.items() if k != "state_distribution"}))
    print(era.to_string(index=False))


if __name__ == "__main__":
    main()
