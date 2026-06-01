"""
Fuzzy Inference Engine for AI-Augmented Software Development DSS
=================================================================

Implements the Mamdani-style Fuzzy Inference System described in:
  Suduc et al. "A Fuzzy Hybrid Decision Support System (DSS) to Govern
  Non-linear Dynamics and Uncertainty in AI-Augmented Software
  Development." Journal of Systems & Software (under revision, 2026).

All membership-function parameters, the 12-rule base (Table VI of the
manuscript), and the Centre-of-Gravity defuzzification (Eq. 5) are
reproduced verbatim here. This module is the *single source of truth*:
the paper's Figures 2, 3, 6 and 7 are regenerated from the constants
defined below.

Inputs
------
- DSPD : Delivered Story Points per Deployment        (Volume,    [0, 50])
- LTBF : Lead Time for Business Features (days)       (Value,     [0, 30])
- Qd   : Quality Duo = AI Acceptance Rate / Code Churn(Viability, [0, 10])

Output
------
- PHS  : Project Health Score                         ([0, 100])
- linguistic_state : one of {Critical Risk, At Risk, Sustainable,
                              High Performance, Elite AI}
- rule_activations : per-rule firing strength (transparency)
- input_memberships : degree of membership of each input in each set
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np
import skfuzzy as fuzz


# ---------------------------------------------------------------------------
# Universe of discourse (continuous domains for each variable)
# ---------------------------------------------------------------------------
# Resolution chosen so that defuzzification error < 0.1 PHS points.
DSPD_UNIVERSE = np.arange(0, 50.01, 0.1)
LTBF_UNIVERSE = np.arange(0, 30.01, 0.1)
QD_UNIVERSE   = np.arange(0, 10.01, 0.01)
PHS_UNIVERSE  = np.arange(0, 100.01, 0.1)


# ---------------------------------------------------------------------------
# Membership function parameters (Tables II, IV, VI, and 3 of the paper)
# ---------------------------------------------------------------------------
# Format: trapezoid = [a, b, c, d] (4 points); triangle = [a, b, c] (3 points).

DSPD_PARAMS: Dict[str, List[float]] = {
    "low":     [0,  0,  5,  10],   # Stagnant  (trapezoid)  — Table II
    "average": [8,  12, 16],       # Steady    (triangle)
    "high":    [14, 20, 50, 50],   # AI-Accel. (trapezoid)
}

LTBF_PARAMS: Dict[str, List[float]] = {
    "rapid":    [0, 0,  2,  5],    # Elite AI  (trapezoid)  — Table IV
    "nominal":  [4, 7,  10],       # Human     (triangle)
    "sluggish": [9, 15, 30, 30],   # AI-Tax    (trapezoid)
}

QD_PARAMS: Dict[str, List[float]] = {
    "fragile":   [0,   0, 0.8, 1.5],   # Table VI
    "stable":    [1.2, 3, 5],
    "resilient": [4.5, 7, 10, 10],
}

PHS_PARAMS: Dict[str, List[float]] = {
    "critical_risk":     [0,  0,    10,  25],    # 3 / Fig. 7
    "at_risk":           [20, 35,   50],
    "sustainable":       [45, 60,   75],
    "high_performance":  [60, 72.5, 85],
    "elite_ai":          [70, 85,   100, 100],
}

# Friendly labels used in API responses & UI
PHS_LABELS = {
    "critical_risk":    "Critical Risk",
    "at_risk":          "At Risk",
    "sustainable":      "Sustainable",
    "high_performance": "High Performance",
    "elite_ai":         "Elite AI Maturity",
}


# ---------------------------------------------------------------------------
# Rule base (Table VI — 12 rules, Mamdani-style)
# ---------------------------------------------------------------------------
# Each rule: (DSPD label, LTBF label, Qd label, PHS label, rationale)
RULES: List[Tuple[str, str, str, str, str]] = [
    ("high",    "rapid",    "resilient", "elite_ai",         "R1: Maximum synergy; AI dividend fully realised."),
    ("high",    "rapid",    "stable",    "high_performance", "R2: Fast & stable, missing elite refactoring."),
    ("high",    "sluggish", "fragile",   "critical_risk",    "R3: Technical Debt Spiral (high noise, low quality)."),
    ("high",    "nominal",  "stable",    "sustainable",      "R4: Solid AI augmentation with manageable overhead."),
    ("average", "nominal",  "stable",    "sustainable",      "R5: High-performing human team, balanced."),
    ("average", "sluggish", "fragile",   "at_risk",          "R6: Standard volume but AI-induced churn."),
    ("average", "rapid",    "stable",    "sustainable",      "R7: Above-average efficiency for human-led team."),
    ("low",     "sluggish", "fragile",   "critical_risk",    "R8: Failure state; AI causing friction."),
    ("low",     "nominal",  "stable",    "at_risk",          "R9: Safe but underperforming vs. industry medians."),
    ("high",    "sluggish", "stable",    "at_risk",          "R10: Volume high, AI-Tax slowing reviews."),
    ("average", "nominal",  "resilient", "high_performance", "R11: Exceptional stability despite avg throughput."),
    ("high",    "rapid",    "fragile",   "critical_risk",    "R12: Fake Speed; prone to collapse."),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mf(universe: np.ndarray, params: List[float]) -> np.ndarray:
    """Build a trapezoidal (4 params) or triangular (3 params) membership function."""
    if len(params) == 4:
        return fuzz.trapmf(universe, params)
    if len(params) == 3:
        return fuzz.trimf(universe, params)
    raise ValueError(f"Membership function expects 3 or 4 params, got {len(params)}")


def _build_mfs(universe: np.ndarray, params: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    return {name: _mf(universe, p) for name, p in params.items()}


# Pre-computed once at import time
_MFS_DSPD = _build_mfs(DSPD_UNIVERSE, DSPD_PARAMS)
_MFS_LTBF = _build_mfs(LTBF_UNIVERSE, LTBF_PARAMS)
_MFS_QD   = _build_mfs(QD_UNIVERSE,   QD_PARAMS)
_MFS_PHS  = _build_mfs(PHS_UNIVERSE,  PHS_PARAMS)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    """Full transparency record of one fuzzy inference run."""
    inputs: Dict[str, float]
    phs: float
    linguistic_state: str
    state_membership: float                       # μ of the dominant output set
    input_memberships: Dict[str, Dict[str, float]]
    rule_activations: List[Dict]                  # firing strength per rule

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(dspd: float, ltbf: float, qd: float) -> InferenceResult:
    """
    Run one Mamdani inference cycle.

    Parameters
    ----------
    dspd : Delivered Story Points per Deployment   (clamped to [0, 50])
    ltbf : Lead Time for Business Features (days)  (clamped to [0, 30])
    qd   : Quality Duo                              (clamped to [0, 10])
    """
    # --- 1. Clamp to universes (defensive; reviewers hate NaN) -----------
    dspd_c = float(np.clip(dspd, DSPD_UNIVERSE[0], DSPD_UNIVERSE[-1]))
    ltbf_c = float(np.clip(ltbf, LTBF_UNIVERSE[0], LTBF_UNIVERSE[-1]))
    qd_c   = float(np.clip(qd,   QD_UNIVERSE[0],   QD_UNIVERSE[-1]))

    # --- 2. Fuzzification: degree of membership for each input -----------
    mu_dspd = {k: float(fuzz.interp_membership(DSPD_UNIVERSE, _MFS_DSPD[k], dspd_c))
               for k in DSPD_PARAMS}
    mu_ltbf = {k: float(fuzz.interp_membership(LTBF_UNIVERSE, _MFS_LTBF[k], ltbf_c))
               for k in LTBF_PARAMS}
    mu_qd   = {k: float(fuzz.interp_membership(QD_UNIVERSE,   _MFS_QD[k],   qd_c))
               for k in QD_PARAMS}

    # --- 3. Rule evaluation: firing strength = min(μA, μB, μC) -----------
    aggregated = np.zeros_like(PHS_UNIVERSE)
    rule_log: List[Dict] = []

    for idx, (d_lbl, l_lbl, q_lbl, out_lbl, rationale) in enumerate(RULES, start=1):
        firing = min(mu_dspd[d_lbl], mu_ltbf[l_lbl], mu_qd[q_lbl])
        # Mamdani implication: clip the consequent MF at the firing strength
        clipped = np.fmin(firing, _MFS_PHS[out_lbl])
        aggregated = np.fmax(aggregated, clipped)

        rule_log.append({
            "id":            f"R{idx}",
            "antecedents":   {"DSPD": d_lbl, "LTBF": l_lbl, "Qd": q_lbl},
            "consequent":    out_lbl,
            "firing_strength": float(firing),
            "rationale":     rationale,
        })

    # --- 4. Defuzzification: Centre of Gravity (Eq. 5) -------------------
    if aggregated.sum() == 0:
        # Edge case: no rule fires (input outside any antecedent support).
        # Fall back to neutral midpoint and flag it.
        phs = 50.0
        dominant = "sustainable"
        dominant_mu = 0.0
    else:
        phs = float(fuzz.defuzz(PHS_UNIVERSE, aggregated, "centroid"))
        # Determine which output set is dominant at the crisp PHS
        memberships_at_phs = {
            k: float(fuzz.interp_membership(PHS_UNIVERSE, _MFS_PHS[k], phs))
            for k in PHS_PARAMS
        }
        dominant = max(memberships_at_phs, key=memberships_at_phs.get)
        dominant_mu = memberships_at_phs[dominant]

    return InferenceResult(
        inputs={"DSPD": dspd_c, "LTBF": ltbf_c, "Qd": qd_c},
        phs=round(phs, 2),
        linguistic_state=PHS_LABELS[dominant],
        state_membership=round(dominant_mu, 4),
        input_memberships={
            "DSPD": {k: round(v, 4) for k, v in mu_dspd.items()},
            "LTBF": {k: round(v, 4) for k, v in mu_ltbf.items()},
            "Qd":   {k: round(v, 4) for k, v in mu_qd.items()},
        },
        rule_activations=rule_log,
    )


# ---------------------------------------------------------------------------
# Convenience: expose the universes & MF curves for plotting in the UI
# ---------------------------------------------------------------------------
def membership_curves() -> Dict[str, Dict]:
    """Return the (x, y) curves for every variable's membership functions."""
    def _pack(universe: np.ndarray, mfs: Dict[str, np.ndarray]) -> Dict:
        return {
            "x": universe.round(3).tolist(),
            "sets": {k: v.round(4).tolist() for k, v in mfs.items()},
        }
    return {
        "DSPD": _pack(DSPD_UNIVERSE, _MFS_DSPD),
        "LTBF": _pack(LTBF_UNIVERSE, _MFS_LTBF),
        "Qd":   _pack(QD_UNIVERSE,   _MFS_QD),
        "PHS":  _pack(PHS_UNIVERSE,  _MFS_PHS),
    }
