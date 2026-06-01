"""
Run the fuzzy DSS over the quarterly metrics, then compute the statistics
that the OSS longitudinal study needs to answer Reviewer #1's questions.

Outputs (all in `results/`):
  - phs_timeseries.csv         per (project, quarter): inputs + PHS + state
  - era_comparison.csv         per project: pre-AI vs AI-era stats + tests
  - discriminative_power.json  variance, entropy, cross-project Kruskal–Wallis
  - REPORT.md                  human-readable summary; the rebuttal letter
                               and the new 5 of the paper draw from this

Run:
    python src/run_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# Path setup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fuzzy_engine import infer  # noqa: E402

CONFIG = ROOT / "config" / "projects.yaml"
DERIVED = ROOT / "data" / "derived" / "quarterly_metrics.csv"
SAMPLE  = ROOT / "data" / "sample"  / "quarterly_metrics.csv"
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_metrics() -> tuple[pd.DataFrame, str]:
    """Prefer derived (live-fetched) data; fall back to bundled sample."""
    if DERIVED.exists():
        return pd.read_csv(DERIVED), "live"
    return pd.read_csv(SAMPLE), "sample"


def load_eras() -> dict[str, list[str]]:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    return {
        era: meta["quarters"]
        for era, meta in cfg["eras"].items()
    }


# ---------------------------------------------------------------------------
# Apply DSS
# ---------------------------------------------------------------------------
def apply_dss(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        result = infer(r["dspd"], r["ltbf"], r["qd"])
        # Identify which rule(s) fired (for diagnostics)
        fired = [a["id"] for a in result.rule_activations if a["firing_strength"] > 0]
        rows.append({
            **r.to_dict(),
            "phs":              result.phs,
            "linguistic_state": result.linguistic_state,
            "rules_fired":      ",".join(fired) if fired else "NONE",
            "n_rules_fired":    len(fired),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Era comparison: pre-AI vs AI-era per project
# ---------------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta — non-parametric effect size in [-1, 1]."""
    a, b = np.asarray(a), np.asarray(b)
    n_gt = sum((x > y) for x in a for y in b)
    n_lt = sum((x < y) for x in a for y in b)
    return (n_gt - n_lt) / (len(a) * len(b))


def era_comparison(df: pd.DataFrame, eras: dict) -> pd.DataFrame:
    pre = set(eras["pre_ai"])
    ai  = set(eras["ai_era"])

    rows = []
    for project_id, sub in df.groupby("project_id"):
        pre_phs = sub.loc[sub["quarter"].isin(pre), "phs"].values
        ai_phs  = sub.loc[sub["quarter"].isin(ai),  "phs"].values

        if len(pre_phs) < 2 or len(ai_phs) < 2:
            rows.append({"project_id": project_id, "note": "insufficient data"})
            continue

        u, p = stats.mannwhitneyu(ai_phs, pre_phs, alternative="two-sided")
        delta = cliffs_delta(ai_phs, pre_phs)

        rows.append({
            "project_id":      project_id,
            "n_pre":           len(pre_phs),
            "n_ai":            len(ai_phs),
            "phs_pre_mean":    round(float(np.mean(pre_phs)), 2),
            "phs_pre_median":  round(float(np.median(pre_phs)), 2),
            "phs_ai_mean":     round(float(np.mean(ai_phs)), 2),
            "phs_ai_median":   round(float(np.median(ai_phs)), 2),
            "phs_shift":       round(float(np.mean(ai_phs) - np.mean(pre_phs)), 2),
            "mwu_U":           round(float(u), 2),
            "mwu_p":           round(float(p), 4),
            "cliffs_delta":    round(float(delta), 3),
            "significant_05":  bool(p < 0.05),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Discriminative power
# ---------------------------------------------------------------------------
def discriminative_power(df: pd.DataFrame) -> dict:
    """
    Discriminative power = does the DSS produce different outputs for different
    inputs? Three measures:
      1. Spread of PHS across all observations (variance, IQR).
      2. Entropy of the linguistic-state distribution (Shannon).
      3. Cross-project Kruskal-Wallis on PHS (do projects differ from each
         other beyond chance?).
    """
    phs = df["phs"].values
    states = df["linguistic_state"].values

    # 1. Spread
    var = float(np.var(phs))
    iqr = float(np.percentile(phs, 75) - np.percentile(phs, 25))

    # 2. Entropy
    counts = Counter(states)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    entropy_bits = float(-(probs * np.log2(probs)).sum())
    entropy_max  = float(np.log2(5))   # 5 possible linguistic states
    entropy_norm = entropy_bits / entropy_max if entropy_max > 0 else 0.0

    # 3. Cross-project Kruskal-Wallis
    groups = [g["phs"].values for _, g in df.groupby("project_id")]
    h, p = stats.kruskal(*groups)

    # 4. % of project-quarters that fell into the no-rule fallback
    fallback_pct = float((df["rules_fired"] == "NONE").sum()) / len(df) * 100

    return {
        "n_observations":         len(df),
        "phs_mean":               round(float(np.mean(phs)), 2),
        "phs_std":                round(float(np.std(phs)),  2),
        "phs_variance":           round(var, 2),
        "phs_iqr":                round(iqr, 2),
        "phs_min":                round(float(np.min(phs)),  2),
        "phs_max":                round(float(np.max(phs)),  2),
        "state_distribution":     dict(counts),
        "entropy_bits":           round(entropy_bits, 3),
        "entropy_normalised":     round(entropy_norm, 3),
        "kruskal_H":              round(float(h), 2),
        "kruskal_p":              round(float(p), 6),
        "kruskal_significant_05": bool(p < 0.05),
        "no_rule_fallback_pct":   round(fallback_pct, 2),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def build_report(source: str, df: pd.DataFrame, era_df: pd.DataFrame, dp: dict) -> str:
    lines = []
    lines.append("# OSS Longitudinal Study — Findings\n")
    lines.append(f"_Auto-generated by `src/run_analysis.py`. Data source: **{source}**._\n")
    lines.append("")

    # --- Executive summary ------------------------------------------------
    lines.append("## Executive summary\n")
    sig_projects = era_df[era_df["significant_05"] == True]["project_id"].tolist()
    lines.append(
        f"- **{dp['n_observations']}** project-quarters analysed across "
        f"**{df['project_id'].nunique()}** OSS projects.\n"
        f"- PHS distribution: mean = **{dp['phs_mean']}**, "
        f"std = **{dp['phs_std']}**, range = "
        f"**[{dp['phs_min']}, {dp['phs_max']}]**.\n"
        f"- Linguistic-state entropy: "
        f"**{dp['entropy_bits']} bits** "
        f"(normalised {dp['entropy_normalised']} of max 1.0).\n"
        f"- Cross-project Kruskal-Wallis: H = **{dp['kruskal_H']}**, "
        f"p = **{dp['kruskal_p']}** "
        f"({'projects differ significantly' if dp['kruskal_significant_05'] else 'projects do NOT differ significantly'}).\n"
        f"- Pre-AI vs AI-era shift significant in **{len(sig_projects)} of "
        f"{len(era_df)}** projects: {', '.join(sig_projects) if sig_projects else 'none'}.\n"
        f"- No-rule fallback fired in **{dp['no_rule_fallback_pct']}%** of "
        f"project-quarters (rule-base coverage indicator).\n"
    )

    # --- Discriminative power ---------------------------------------------
    lines.append("\n## 1. Discriminative power\n")
    lines.append("State distribution across all project-quarters:\n")
    lines.append("| State | Count | Share |")
    lines.append("|---|---:|---:|")
    total = sum(dp["state_distribution"].values())
    for state, c in sorted(dp["state_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"| {state} | {c} | {c/total*100:.1f}% |")
    lines.append("")
    if dp["kruskal_significant_05"]:
        lines.append(
            f"Kruskal-Wallis test on PHS across the six projects yields "
            f"H = **{dp['kruskal_H']}**, p = **{dp['kruskal_p']}** — projects "
            f"are statistically distinguishable. The DSS therefore **has** "
            f"discriminative power across this sample.\n"
        )
    else:
        lines.append(
            f"Kruskal-Wallis test on PHS yields p = **{dp['kruskal_p']}** "
            f"(> 0.05) — across this sample, the DSS does **not** "
            f"distinguish projects beyond chance. This is a finding worth "
            f"discussing in the threats-to-validity section.\n"
        )

    # --- Era comparison ---------------------------------------------------
    lines.append("\n## 2. Pre-AI vs AI-era shift, per project\n")
    lines.append("Mann-Whitney U is non-parametric; Cliff's delta is a "
                 "non-parametric effect size in [-1, 1].\n")
    lines.append("| Project | n_pre | n_ai | PHS pre (mean) | PHS AI (mean) | "
                 "Δ | MWU U | p | Cliff's δ | Sig. (p<0.05) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for _, r in era_df.iterrows():
        if "note" in r and pd.notna(r.get("note")):
            continue
        sig = "✓" if r["significant_05"] else "—"
        lines.append(
            f"| {r['project_id']} | {r['n_pre']} | {r['n_ai']} | "
            f"{r['phs_pre_mean']} | {r['phs_ai_mean']} | "
            f"{r['phs_shift']:+.2f} | {r['mwu_U']} | {r['mwu_p']} | "
            f"{r['cliffs_delta']:+.3f} | {sig} |"
        )

    # --- Per-project narrative -------------------------------------------
    lines.append("\n## 3. Per-project trajectories\n")
    for project_id, sub in df.groupby("project_id"):
        sub = sub.sort_values("quarter")
        states_q = sub["linguistic_state"].value_counts()
        first_state = sub["linguistic_state"].iloc[0]
        last_state  = sub["linguistic_state"].iloc[-1]
        lines.append(f"### `{project_id}`\n")
        lines.append(
            f"- Quarters: **{len(sub)}** "
            f"({sub['quarter'].iloc[0]} → {sub['quarter'].iloc[-1]})\n"
            f"- First-quarter state: **{first_state}**, "
            f"last-quarter state: **{last_state}**\n"
            f"- State distribution: "
            f"{', '.join(f'{s} ({c})' for s, c in states_q.items())}\n"
            f"- PHS range: **[{sub['phs'].min():.1f}, {sub['phs'].max():.1f}]**, "
            f"std = **{sub['phs'].std():.2f}**\n"
        )

    # --- Limitations & caveats -------------------------------------------
    lines.append("\n## 4. Limitations and threats to validity\n")
    lines.append(
        "1. **Proxy fidelity.** DSPD, LTBF, and Qd are computed from public "
        "GitHub signals (PR counts, cycle times, churn ratios) rather than "
        "the internal telemetry the manuscript's narrative assumes. "
        "Practitioner-side metrics (e.g. true AI-acceptance from IDE plugins) "
        "would refine Qd in particular. See `docs/METHODOLOGY.md` 3.\n"
        "2. **Era boundaries are assumed, not measured.** AI-tooling adoption "
        "diffuses gradually inside each project. The era partitioning in "
        "`config/projects.yaml` is calibrated against documented public "
        "milestones, not per-project adoption curves.\n"
        "3. **Selection effect.** The six projects are large, popular, and "
        "well-instrumented. Generalisation to smaller or less observable "
        "projects is not claimed.\n"
        "4. **Rule-base coverage.** A non-trivial fraction of project-"
        "quarters fell into the no-rule fallback "
        f"({dp['no_rule_fallback_pct']}%). This is a separate diagnostic "
        "for the rule-base completeness and is documented in the parent "
        "fuzzy-dss repository.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df_in, source = load_metrics()
    print(f"Loaded {len(df_in)} project-quarters from {source} dataset.")

    print("Applying fuzzy DSS to every row...")
    df = apply_dss(df_in)
    df.to_csv(RESULTS / "phs_timeseries.csv", index=False)
    print(f"  → {RESULTS / 'phs_timeseries.csv'}")

    print("Computing pre-AI vs AI-era comparison...")
    eras = load_eras()
    era_df = era_comparison(df, eras)
    era_df.to_csv(RESULTS / "era_comparison.csv", index=False)
    print(f"  → {RESULTS / 'era_comparison.csv'}")

    print("Computing discriminative power...")
    dp = discriminative_power(df)
    with open(RESULTS / "discriminative_power.json", "w") as f:
        json.dump(dp, f, indent=2)
    print(f"  → {RESULTS / 'discriminative_power.json'}")

    print("Building human-readable report...")
    report = build_report(source, df, era_df, dp)
    with open(RESULTS / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  → {RESULTS / 'REPORT.md'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
