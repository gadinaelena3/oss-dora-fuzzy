"""
Sensitivity analysis for the scaling factors k_DSPD, k_LTBF, k_Qd
=================================================================

This script reproduces Table XI of the Addendum (Scaling Factor
Calibration & Sensitivity Analysis).

For each candidate value of k on each input axis, the proxy is recomputed
at every (project, quarter) pair in the study, every observation is
assigned to the fuzzy set in which it has the highest membership grade
(*peak-membership classification*) against the actual fuzzy sets of
Section 3, and the Shannon entropy of the resulting three-zone
distribution is computed.

The k that maximises the entropy is the scaling factor that distributes
the 144 project-quarters most evenly across the three linguistic zones
on that axis. This is the operational definition of discriminative power
on a single input axis.

Two candidate grids are defined per axis:

  * ``KS_FULL[axis]`` -- the wider exploratory grid actually swept by this
    script. It is a *superset* of the published grid and includes finer
    candidates used during calibration.
  * ``KS_TABLE_XI[axis]`` -- the exact subset of candidates printed in
    Table XI of the Addendum.

The script sweeps the full grid, then prints the Table-XI subset so that
the printed output matches the published table verbatim while the argmax
is taken over the full grid. The selected optima (k_DSPD = 14,
k_LTBF = 8, k_Qd = 3) are identical under both grids.

Inputs
------
    results/phs_timeseries.csv  -- 144 rows, must contain the columns:
        project_id, n_prs, median_cycle_days, churn_ratio,
        first_time_merge_rate

Outputs
-------
    Prints three sweep tables (DSPD, LTBF, Qd) and the argmax per axis,
    matching Table XI of the Addendum.

Dependencies
------------
    numpy, pandas, scikit-fuzzy
        pip install numpy pandas scikit-fuzzy

Usage
-----
    python sensitivity_analysis.py [path/to/phs_timeseries.csv]

The membership-function parameters below are copied verbatim from
src/fuzzy_engine.py (Tables II, IV, VI of the manuscript).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz


# ---------------------------------------------------------------------------
# Fuzzy sets (Tables II, IV, VI; identical to src/fuzzy_engine.py)
# ---------------------------------------------------------------------------
# Universe step sizes (0.1 for DSPD/LTBF, 0.01 for Qd) are discretisation
# choices for the membership-function lookup grid; they do not affect the
# peak-membership classification at the resolution used here.
DSPD_UNIVERSE = np.arange(0, 50.01, 0.1)
LTBF_UNIVERSE = np.arange(0, 30.01, 0.1)
QD_UNIVERSE   = np.arange(0, 10.01, 0.01)

DSPD_PARAMS = {
    "Low":     [0,  0,  5,  10],   # trapezoid
    "Average": [8,  12, 16],       # triangle
    "High":    [14, 20, 50, 50],   # trapezoid
}
LTBF_PARAMS = {
    "Rapid":    [0, 0,  2,  5],    # trapezoid
    "Nominal":  [4, 7,  10],       # triangle
    "Sluggish": [9, 15, 30, 30],   # trapezoid
}
QD_PARAMS = {
    "Fragile":   [0,   0, 0.8, 1.5],   # trapezoid
    "Stable":    [1.2, 3, 5],          # triangle
    "Resilient": [4.5, 7, 10, 10],     # trapezoid
}

LABELS = {
    "DSPD": ["Low", "Average", "High"],
    "LTBF": ["Rapid", "Nominal", "Sluggish"],
    "Qd":   ["Fragile", "Stable", "Resilient"],
}

# Churn floor used in the Qd proxy: the raw churn ratio is clamped to a
# minimum of 0.05 before scaling, to guard against division by a
# near-zero churn denominator. On the study dataset the minimum observed
# churn ratio is ~0.078, so this floor is never active and the proxy is
# numerically identical to the plain ``FTMR * 10 / (churn * k)`` form
# stated in Table XI of the Addendum. It is documented here and in the
# Addendum so that code and paper agree exactly.
CHURN_FLOOR = 0.05


def _mf(universe: np.ndarray, params: list[float]) -> np.ndarray:
    """Build a trapezoidal (4 params) or triangular (3 params) MF."""
    if len(params) == 4:
        return fuzz.trapmf(universe, params)
    if len(params) == 3:
        return fuzz.trimf(universe, params)
    raise ValueError(f"Expected 3 or 4 params, got {len(params)}")


def _build(universe, param_dict):
    return {name: _mf(universe, p) for name, p in param_dict.items()}


# Pre-compute the membership-function curves once
MFS = {
    "DSPD": _build(DSPD_UNIVERSE, DSPD_PARAMS),
    "LTBF": _build(LTBF_UNIVERSE, LTBF_PARAMS),
    "Qd":   _build(QD_UNIVERSE,   QD_PARAMS),
}
UNIVERSES = {"DSPD": DSPD_UNIVERSE, "LTBF": LTBF_UNIVERSE, "Qd": QD_UNIVERSE}


# ---------------------------------------------------------------------------
# Candidate k grids
# ---------------------------------------------------------------------------
# Wider exploratory grid actually swept (superset of the published table).
KS_FULL = {
    "DSPD": [8, 10, 12, 14, 15, 16, 18, 20, 25, 30, 35],
    "LTBF": [2, 3, 4, 5, 6, 7, 8, 10, 12],
    "Qd":   [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8],
}
# Exact subset printed in Table XI of the Addendum.
KS_TABLE_XI = {
    "DSPD": [10, 14, 15, 20, 25, 30, 35],
    "LTBF": [2, 3, 4, 5, 6, 8, 10, 12],
    "Qd":   [1, 2, 3, 4, 5, 6],
}


# ---------------------------------------------------------------------------
# Peak-membership classification and Shannon entropy
# ---------------------------------------------------------------------------
def classify(axis: str, value: float) -> str:
    """Return the label of the fuzzy set with the highest membership."""
    universe = UNIVERSES[axis]
    mus = {
        lbl: float(fuzz.interp_membership(universe, MFS[axis][lbl], value))
        for lbl in LABELS[axis]
    }
    # Tie-break in the natural label order: Low<Average<High,
    # Rapid<Nominal<Sluggish, Fragile<Stable<Resilient.
    best = LABELS[axis][0]
    for lbl in LABELS[axis]:
        if mus[lbl] > mus[best]:
            best = lbl
    return best


def shannon_entropy(counts: dict[str, int]) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def sweep(axis: str, scale_fn, ks: list[float]) -> list[tuple]:
    """Return [(k, counts_dict, entropy), ...] for each candidate k."""
    out = []
    for k in ks:
        scaled = scale_fn(k)
        counts = {lbl: 0 for lbl in LABELS[axis]}
        for v in scaled:
            counts[classify(axis, v)] += 1
        out.append((k, counts, shannon_entropy(counts)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} project-quarters from {csv_path}")

    # Per-project mean PR count (used in the DSPD proxy)
    df["project_mean_prs"] = df.groupby("project_id")["n_prs"].transform("mean")

    # Proxy formulas (identical to src/compute_metrics.py)
    def dspd_at(k):
        return np.clip(df["n_prs"] / df["project_mean_prs"] * k, 0, 50).values

    def ltbf_at(k):
        return np.clip(df["median_cycle_days"] * k, 0, 30).values

    def qd_at(k):
        return np.clip(
            df["first_time_merge_rate"] * 10
            / (df["churn_ratio"].clip(lower=CHURN_FLOOR) * k),
            0, 10,
        ).values

    sweeps = [
        ("DSPD", "n_prs / mean_prs * k",               dspd_at),
        ("LTBF", "median_cycle_days * k",              ltbf_at),
        ("Qd",   "FTMR * 10 / (max(churn, 0.05) * k)", qd_at),
    ]

    for axis, formula, fn in sweeps:
        print()
        print(f"=== {axis}: clip({formula}, ...) ===")
        header = f"{'k':>5} | " + " ".join(f"{l:>10}" for l in LABELS[axis]) \
                 + " | Entropy (bits)"
        print(header)
        print("-" * len(header))

        # Sweep the full grid, then print only the published Table XI rows.
        rows_full = sweep(axis, fn, KS_FULL[axis])
        rows_by_k = {k: (counts, h) for k, counts, h in rows_full}
        for k in KS_TABLE_XI[axis]:
            counts, h = rows_by_k[k]
            counts_str = " ".join(f"{counts[l]:>10}" for l in LABELS[axis])
            print(f"{k:>5} | {counts_str} | {h:>13.4f}")

        # Argmax over the full grid.
        k_best, counts_best, h_best = max(rows_full, key=lambda r: r[2])
        print(f"  -> argmax: k = {k_best}, H = {h_best:.4f} "
              f"(distribution {counts_best})")

    print(f"\nMaximum possible entropy for 3 equiprobable zones: "
          f"log2(3) = {math.log2(3):.4f} bits")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path("results/phs_timeseries.csv")

    if not path.exists():
        print(f"Error: The file was not found at {path}")
        sys.exit(1)

    raise SystemExit(main(path))
