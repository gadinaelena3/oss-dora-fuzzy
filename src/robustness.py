import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import compute_metrics as cm
from fuzzy_engine import infer

OUT = ROOT / "results" / "robustness"
ALPHA = 0.05 / 6
GRID = {"k_dspd": [8, 10, None, 14, 16], "k_ltbf": [5, None, 9, 11], "k_qd": [3, None, 6]}
SWEEP = {"k_dspd": [8, 10, None, 14, 16, 20, 25], "k_ltbf": [3, 4, 5, 6, None, 8, 10, 12], "k_qd": [2, 3, 4, None, 5, 6]}
SETS = {
    "dspd": {"Low": (0, 0, 5, 10), "Average": (8, 12, 12, 16), "High": (14, 20, 50, 50)},
    "ltbf": {"Rapid": (0, 0, 2, 5), "Nominal": (4, 7, 7, 10), "Sluggish": (9, 15, 30, 30)},
    "qd": {"Fragile": (0, 0, 0.8, 1.5), "Stable": (1.2, 3, 3, 5), "Resilient": (4.5, 7, 10, 10)},
}


def era(q):
    y, n = int(q[:4]), int(q[-1])
    if y <= 2020:
        return "pre"
    if (y == 2022 and n == 4) or y >= 2023:
        return "ai"
    return "transition"


def trap(x, a, b, c, d):
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def zone(axis, x):
    m = {name: trap(x, *p) for name, p in SETS[axis].items()}
    return max(m, key=m.get)


def run_dss(m):
    res = [infer(a, b, c) for a, b, c in zip(m.dspd, m.ltbf, m.qd)]
    out = m.copy()
    out["phs"] = [r.phs for r in res]
    out["state"] = [r.linguistic_state for r in res]
    out["fallback"] = [max(x["firing_strength"] for x in r.rule_activations) == 0 for r in res]
    out["era"] = out.quarter.map(era)
    return out


def era_tests(d):
    rows = {}
    for p, x in d.groupby("project_id"):
        pre, ai = x[x.era == "pre"].phs.values, x[x.era == "ai"].phs.values
        u, pv = stats.mannwhitneyu(ai, pre, alternative="two-sided")
        rows[p] = {"pre_mean": round(pre.mean(), 2), "ai_mean": round(ai.mean(), 2),
                   "shift": round(ai.mean() - pre.mean(), 2), "U": u, "p": round(pv, 4),
                   "delta": round(2 * u / (len(pre) * len(ai)) - 1, 3)}
    return rows


def summary(d):
    h, p = stats.kruskal(*[x.phs.values for _, x in d.groupby("project_id")])
    return {"kruskal_H": round(h, 2), "kruskal_p": p, "fallback_pct": round(100 * d.fallback.mean(), 2),
            "states": d.state.value_counts().to_dict()}


def flat(k, d):
    row = dict(k, **{a: b for a, b in summary(d).items() if a != "states"})
    for p, t in era_tests(d).items():
        row[f"{p}_delta"], row[f"{p}_p"] = t["delta"], t["p"]
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prs = cm.load_prs()
    s = cm.signals(prs)
    k0, base = cm.calibrate(s)
    ref = run_dss(cm.proxies(s, base, **k0))
    projects = sorted(s.project_id.unique())

    rows = []
    for axis, cands in SWEEP.items():
        col = axis[2:]
        for c in cands:
            k = dict(k0, **({axis: c} if c is not None else {}))
            z = cm.proxies(s, base, **k)[col].map(lambda v: zone(col, v)).value_counts()
            counts = [int(z.get(n, 0)) for n in SETS[col]]
            pr = np.array([c_ for c_ in counts if c_ > 0]) / sum(counts)
            rows.append({"axis": axis, "k": k[axis], "reported": c is None, **dict(zip(["zone1", "zone2", "zone3"], counts)),
                         "entropy_bits": round(float(-(pr * np.log2(pr)).sum()), 3)})
    pd.DataFrame(rows).to_csv(OUT / "entropy_sweep.csv", index=False)

    rows = []
    for combo in itertools.product(*GRID.values()):
        k = {a: (k0[a] if v is None else v) for a, v in zip(GRID, combo)}
        rows.append(flat(k, run_dss(cm.proxies(s, base, **k))))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid.csv", index=False)
    gs = []
    for p in projects:
        dl, pv = grid[f"{p}_delta"], grid[f"{p}_p"]
        gs.append({"project": p, "delta_reported": era_tests(ref)[p]["delta"], "delta_min": dl.min(), "delta_max": dl.max(),
                   "n_positive": int((dl > 0).sum()), "n_negative": int((dl < 0).sum()),
                   "n_significant": int((pv < ALPHA).sum()), "n_settings": len(grid)})
    pd.DataFrame(gs).to_csv(OUT / "grid_summary.csv", index=False)

    sb = cm.signals(prs, include_bots=True)
    kb, baseb = cm.calibrate(sb)
    refb = run_dss(cm.proxies(sb, baseb, **kb))
    rows = []
    for col, label in [("dspd", "DSPD proxy"), ("ltbf", "LTBF proxy"), ("qd", "Qd proxy"), ("phs", "PHS")]:
        h1, p1 = stats.kruskal(*[x[col].values for _, x in ref.groupby("project_id")])
        h2, p2 = stats.kruskal(*[x[col].values for _, x in refb.groupby("project_id")])
        rows.append({"variable": label, "H": round(h1, 2), "p": p1, "H_bots_included": round(h2, 2), "p_bots_included": p2})
    pd.DataFrame(rows).to_csv(OUT / "inputs_kruskal.csv", index=False)
    pd.DataFrame([{"variant": "bots excluded", **flat(k0, ref)}, {"variant": "bots included", **flat(kb, refb)}]).to_csv(OUT / "bots.csv", index=False)

    rows, held = [], []
    pre_q = cm.pre_ai_quarters()
    for p in projects:
        k, _ = cm.calibrate(s[s.project_id != p])
        d = run_dss(cm.proxies(s[s.project_id == p], base, **k))
        held.append(d)
        t = era_tests(d)[p]
        r = ref[ref.project_id == p]
        rows.append({"held_out": p, **k, "delta": t["delta"], "p": t["p"], "shift": t["shift"],
                     "max_abs_phs_change": round(float(np.abs(d.phs.values - r.phs.values).max()), 2),
                     "quarters_state_changed": int((d.state.values != r.state.values).sum())})
    held = pd.concat(held)
    pd.DataFrame(rows).to_csv(OUT / "leave_one_project_out.csv", index=False)
    (OUT / "leave_one_project_out_summary.json").write_text(json.dumps(summary(held), default=float, indent=2))

    prev = run_dss(cm.proxies(sb, sb.groupby("project_id").n_prs.mean(), k_dspd=14, k_ltbf=8, k_qd=3))
    pd.DataFrame([{"variant": "complete data, entropy-based calibration (14, 8, 3; whole-window mean; bots included)", **flat({"k_dspd": 14, "k_ltbf": 8, "k_qd": 3}, prev)},
                  {"variant": "complete data, anchored calibration", **flat(k0, ref)}]).to_csv(OUT / "entropy_calibration.csv", index=False)
    ref.to_csv(OUT / "reference.csv", index=False)
    gone = pd.read_csv(ROOT / "data" / "initial" / "not_returned.csv")
    gone["created"] = pd.to_datetime(gone.created_at).dt.tz_localize(None)
    gone["merged"] = pd.to_datetime(gone.merged_at).dt.tz_localize(None)
    gone["days"] = (gone.merged - gone.created).dt.total_seconds() / 86400
    gone["quarter"] = gone.merged.dt.to_period("Q").astype(str)
    gone = gone.assign(bot=False, default_base=True, from_fork=False, self_merged=False, no_review=False)
    sr = cm.signals(pd.concat([prs, gone[prs.columns]], ignore_index=True))
    kr, baser = cm.calibrate(sr)
    refr = run_dss(cm.proxies(sr, baser, **kr))
    both = ref.merge(refr, on=["project_id", "quarter"], suffixes=("", "_restored"))
    pd.DataFrame([{"variant": "reported", "n_prs": int(s.n_prs.sum()), **flat(k0, ref)},
                  {"variant": "not-returned pull requests restored", "n_prs": int(sr.n_prs.sum()), **flat(kr, refr),
                   "quarters_state_changed": int((both.state != both.state_restored).sum()),
                   "max_abs_phs_change": round(float((both.phs - both.phs_restored).abs().max()), 2)}]).to_csv(OUT / "restored.csv", index=False)
    gone.assign(era=gone.quarter.map(era)).groupby(["project_id", "era"]).size().unstack(fill_value=0).to_csv(OUT / "not_returned_by_era.csv")
    refb.to_csv(OUT / "reference_bots_included.csv", index=False)

    rows = []
    years = {"2019": [f"2019Q{i}" for i in range(1, 5)], "2020": [f"2020Q{i}" for i in range(1, 5)]}
    for cal, test in [("2019", "2020"), ("2020", "2019")]:
        k, b = cm.calibrate(s, years[cal])
        d = run_dss(cm.proxies(s, b, **k))
        row = {"calibrated_on": cal, "pre_ai_test_quarters": test, **k, **{a: v for a, v in summary(d).items() if a != "states"},
               "quarters_state_changed": int((d.state.values != ref.state.values).sum())}
        for p, x in d.groupby("project_id"):
            pre, ai = x[x.quarter.isin(years[test])].phs.values, x[x.era == "ai"].phs.values
            u, pv = stats.mannwhitneyu(ai, pre, alternative="two-sided")
            row[f"{p}_delta"], row[f"{p}_p"] = round(2 * u / (len(pre) * len(ai)) - 1, 3), round(pv, 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "temporal_holdout.csv", index=False)

    main = prs[prs.default_base]
    sm = cm.signals(main)
    km, basem = cm.calibrate(sm)
    refm = run_dss(cm.proxies(sm, basem, **km))
    both = ref.merge(refm, on=["project_id", "quarter"], suffixes=("", "_main"))
    human = prs[~prs.bot]
    share = human.assign(era=human.quarter.map(era)).groupby(["project_id", "era"]).default_base.agg(lambda x: round(100 * (1 - x.mean()), 1)).unstack()
    share["all"] = human.groupby("project_id").default_base.agg(lambda x: round(100 * (1 - x.mean()), 1))
    share["n_other_base"] = human.groupby("project_id").default_base.agg(lambda x: int((~x).sum()))
    share.to_csv(OUT / "other_base_share.csv")
    pre_q = cm.pre_ai_quarters()
    pd.DataFrame([{"variant": "all merged pull requests (reported)", "n_prs": int(s.n_prs.sum()), **flat(k0, ref),
                   "pooled_pre_ai_median_days": round(float(s[s.quarter.isin(pre_q)].median_cycle_days.median()), 3)},
                  {"variant": "default branch only", "n_prs": int(sm.n_prs.sum()), **flat(km, refm),
                   "pooled_pre_ai_median_days": round(float(sm[sm.quarter.isin(pre_q)].median_cycle_days.median()), 3),
                   "quarters_state_changed": int((both.state != both.state_main).sum())}]).to_csv(OUT / "default_branch_only.csv", index=False)
    refm.to_csv(OUT / "reference_default_branch_only.csv", index=False)

    print(json.dumps(k0), json.dumps(summary(ref), default=float))
    print(pd.read_csv(OUT / "temporal_holdout.csv").T.to_string())
    print(pd.read_csv(OUT / "default_branch_only.csv").T.to_string())
    print(pd.read_csv(OUT / "other_base_share.csv").to_string(index=False))
    print(pd.DataFrame(era_tests(ref)).T.to_string())
    print(pd.DataFrame(gs).to_string(index=False))
    print("H range", grid.kruskal_H.min(), grid.kruskal_H.max(), "fallback range", grid.fallback_pct.min(), grid.fallback_pct.max())
    print(pd.read_csv(OUT / "leave_one_project_out.csv").to_string(index=False))
    print(pd.read_csv(OUT / "bots.csv").T.to_string())
    print(pd.read_csv(OUT / "entropy_calibration.csv").T.to_string())
    print(pd.read_csv(OUT / "inputs_kruskal.csv").to_string(index=False))
    print(pd.read_csv(OUT / "restored.csv").T.to_string())
    print(pd.read_csv(OUT / "not_returned_by_era.csv").to_string(index=False))
    print(pd.read_csv(OUT / "entropy_sweep.csv").to_string(index=False))


if __name__ == "__main__":
    main()
