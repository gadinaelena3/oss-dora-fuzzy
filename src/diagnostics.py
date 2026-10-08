import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import compute_metrics as cm
import robustness as rb
from fuzzy_engine import RULES, infer

OUT = ROOT / "results" / "diagnostics.json"
ALPHA = 0.05 / 6
STATE = {"critical_risk": "Critical Risk", "at_risk": "At Risk", "sustainable": "Sustainable",
         "high_performance": "High Performance", "elite_ai": "Elite AI Maturity"}


def exact_p(a, b):
    pooled = np.concatenate([a, b])
    ranks = stats.rankdata(pooled)
    n, total = len(a), len(pooled)
    centre = n * (total + 1) / 2
    obs = abs(ranks[:n].sum() - centre)
    hits = count = 0
    for idx in itertools.combinations(range(total), n):
        count += 1
        hits += abs(ranks[list(idx)].sum() - centre) >= obs - 1e-9
    return hits / count


def u_threshold(n1, n2, alpha):
    counts = Counter()
    for idx in itertools.combinations(range(n1 + n2), n1):
        counts[sum(idx) - n1 * (n1 - 1) // 2] += 1
    total = sum(counts.values())
    for u in range(n1 * n2, n1 * n2 // 2, -1):
        p = 2 * sum(v for k, v in counts.items() if k >= u) / total
        if p >= alpha:
            return round(2 * (u + 1) / (n1 * n2) - 1, 3)
    return None


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    prs = cm.load_prs()
    s = cm.signals(prs)
    k, base = cm.calibrate(s)
    ref = rb.run_dss(cm.proxies(s, base, **k)).sort_values(["project_id", "quarter"]).reset_index(drop=True)
    projects = sorted(ref.project_id.unique())
    out = {}

    out["exact_permutation_p"] = {p: round(exact_p(x[x.era == "ai"].phs.values, x[x.era == "pre"].phs.values), 5) for p, x in ref.groupby("project_id")}
    out["smallest_detectable_abs_delta"] = {"alpha_bonferroni": u_threshold(9, 8, ALPHA), "alpha_0.05": u_threshold(9, 8, 0.05)}
    era_p = [rb.era_tests(ref)[p]["p"] for p in projects]
    out["holm_era_contrasts"] = dict(zip(projects, [round(float(v), 4) for v in holm(np.array(era_p))]))

    out["lag1_autocorrelation"] = {col: {p: (None if x[col].nunique() < 2 else round(float(x[col].autocorr(1)), 2)) for p, x in ref.groupby("project_id")}
                                   for col in ["phs", "n_prs", "dspd", "ltbf", "qd"]}

    pairs, pv = [], []
    for a, b in itertools.combinations(projects, 2):
        pairs.append(f"{a}-{b}")
        pv.append(stats.mannwhitneyu(ref[ref.project_id == a].phs, ref[ref.project_id == b].phs, alternative="two-sided").pvalue)
    adj = holm(np.array(pv))
    out["pairwise_holm"] = {"separated": int((adj < 0.05).sum()), "pairs": len(pairs), "not_separated": [n for n, v in zip(pairs, adj) if v >= 0.05]}

    fired = ref[~ref.fallback]
    h, p = stats.kruskal(*[x.phs.values for _, x in fired.groupby("project_id")])
    out["without_fallback"] = {"n": len(fired), "kruskal_H": round(float(h), 2), "kruskal_p": float(p),
                               "era_tests": {q: {"n_pre": int((x.era == "pre").sum()), "n_ai": int((x.era == "ai").sum()),
                                                 **({} if min((x.era == "pre").sum(), (x.era == "ai").sum()) < 2 else
                                                    dict(zip(["delta", "p"], (lambda u: [round(2 * u.statistic / ((x.era == "pre").sum() * (x.era == "ai").sum()) - 1, 3), round(float(u.pvalue), 4)])(
                                                        stats.mannwhitneyu(x[x.era == "ai"].phs, x[x.era == "pre"].phs, alternative="two-sided")))))}
                                             for q, x in fired.groupby("project_id")}}

    res = [infer(a, b, c) for a, b, c in zip(ref.dspd, ref.ltbf, ref.qd)]
    n_rules = Counter(sum(r["firing_strength"] > 0 for r in x.rule_activations) for x in res)
    values = Counter(round(float(v), 2) for v in ref.phs)
    out["output_granularity"] = {"rules_fired_per_quarter": {str(a): b for a, b in sorted(n_rules.items())},
                                 "most_frequent_scores": {f"{a:.2f}": b for a, b in values.most_common(4)},
                                 "quarters_at_four_most_frequent_scores": sum(b for _, b in values.most_common(4)),
                                 "edge_example": {"dspd_14.00_ltbf_20_qd_3": round(infer(14.0, 20, 3).phs, 2), "dspd_14.05_ltbf_20_qd_3": round(infer(14.05, 20, 3).phs, 2)}}

    human = prs[~prs.bot]
    top = human.groupby(["project_id", "quarter"]).deletions.agg(lambda x: x.max() / max(x.sum(), 1))
    out["qd_proxy"] = {"spearman_ftmr_vs_median_cycle": round(float(stats.spearmanr(ref.first_time_merge_rate, ref.median_cycle_days)[0]), 2),
                       "spearman_qd_vs_deletion_addition_ratio": round(float(stats.spearmanr(ref.qd, ref.churn_ratio)[0]), 2),
                       "quarters_one_pr_over_half_of_deletions": int((top > 0.5).sum()),
                       "largest_single_pr_share": {"project_quarter": "/".join(top.idxmax()), "share_pct": round(100 * float(top.max()), 1)}}

    table = {(a, b, c): STATE[o] for a, b, c, o, _ in RULES}
    crisp = [table.get((rb.zone("dspd", a).lower(), rb.zone("ltbf", b).lower(), rb.zone("qd", c).lower())) for a, b, c in zip(ref.dspd, ref.ltbf, ref.qd)]
    fuzzy = [None if f else st for f, st in zip(ref.fallback, ref.state)]
    out["crisp_lookup"] = {"same_outcome": sum(a == b for a, b in zip(crisp, fuzzy)), "uncovered_crisp": sum(c is None for c in crisp),
                           "uncovered_fuzzy": int(ref.fallback.sum()), "n": len(ref)}

    quarters = sorted(ref.quarter.unique())
    out["ai_era_start"] = {}
    for start in ["2022Q3", "2022Q4", "2023Q1", "2023Q2"]:
        sig = {}
        for q, x in ref.groupby("project_id"):
            u = stats.mannwhitneyu(x[x.quarter >= start].phs, x[x.era == "pre"].phs, alternative="two-sided")
            sig[q] = round(float(u.pvalue), 4)
        out["ai_era_start"][start] = {"p": sig, "below_alpha": [q for q, v in sig.items() if v < ALPHA]}

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
