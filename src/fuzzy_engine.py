from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np
import skfuzzy as fuzz


DSPD_UNIVERSE = np.arange(0, 50.01, 0.1)
LTBF_UNIVERSE = np.arange(0, 30.01, 0.1)
QD_UNIVERSE   = np.arange(0, 10.01, 0.01)
PHS_UNIVERSE  = np.arange(0, 100.01, 0.1)


DSPD_PARAMS: Dict[str, List[float]] = {
    "low":     [0,  0,  5,  10],
    "average": [8,  12, 16],
    "high":    [14, 20, 50, 50],
}

LTBF_PARAMS: Dict[str, List[float]] = {
    "rapid":    [0, 0,  2,  5],
    "nominal":  [4, 7,  10],
    "sluggish": [9, 15, 30, 30],
}

QD_PARAMS: Dict[str, List[float]] = {
    "fragile":   [0,   0, 0.8, 1.5],
    "stable":    [1.2, 3, 5],
    "resilient": [4.5, 7, 10, 10],
}

PHS_PARAMS: Dict[str, List[float]] = {
    "critical_risk":     [0,  0,    10,  25],
    "at_risk":           [20, 35,   50],
    "sustainable":       [45, 60,   75],
    "high_performance":  [60, 72.5, 85],
    "elite_ai":          [70, 85,   100, 100],
}


PHS_LABELS = {
    "critical_risk":    "Critical Risk",
    "at_risk":          "At Risk",
    "sustainable":      "Sustainable",
    "high_performance": "High Performance",
    "elite_ai":         "Elite AI Maturity",
}


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


def _mf(universe: np.ndarray, params: List[float]) -> np.ndarray:
    if len(params) == 4:
        return fuzz.trapmf(universe, params)
    if len(params) == 3:
        return fuzz.trimf(universe, params)
    raise ValueError(f"Membership function expects 3 or 4 params, got {len(params)}")


def _build_mfs(universe: np.ndarray, params: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    return {name: _mf(universe, p) for name, p in params.items()}


_MFS_DSPD = _build_mfs(DSPD_UNIVERSE, DSPD_PARAMS)
_MFS_LTBF = _build_mfs(LTBF_UNIVERSE, LTBF_PARAMS)
_MFS_QD   = _build_mfs(QD_UNIVERSE,   QD_PARAMS)
_MFS_PHS  = _build_mfs(PHS_UNIVERSE,  PHS_PARAMS)


@dataclass
class InferenceResult:
    inputs: Dict[str, float]
    phs: float
    linguistic_state: str
    state_membership: float
    input_memberships: Dict[str, Dict[str, float]]
    rule_activations: List[Dict]

    def to_dict(self) -> dict:
        return asdict(self)


def infer(dspd: float, ltbf: float, qd: float) -> InferenceResult:

    dspd_c = float(np.clip(dspd, DSPD_UNIVERSE[0], DSPD_UNIVERSE[-1]))
    ltbf_c = float(np.clip(ltbf, LTBF_UNIVERSE[0], LTBF_UNIVERSE[-1]))
    qd_c   = float(np.clip(qd,   QD_UNIVERSE[0],   QD_UNIVERSE[-1]))


    mu_dspd = {k: float(fuzz.interp_membership(DSPD_UNIVERSE, _MFS_DSPD[k], dspd_c))
               for k in DSPD_PARAMS}
    mu_ltbf = {k: float(fuzz.interp_membership(LTBF_UNIVERSE, _MFS_LTBF[k], ltbf_c))
               for k in LTBF_PARAMS}
    mu_qd   = {k: float(fuzz.interp_membership(QD_UNIVERSE,   _MFS_QD[k],   qd_c))
               for k in QD_PARAMS}


    aggregated = np.zeros_like(PHS_UNIVERSE)
    rule_log: List[Dict] = []

    for idx, (d_lbl, l_lbl, q_lbl, out_lbl, rationale) in enumerate(RULES, start=1):
        firing = min(mu_dspd[d_lbl], mu_ltbf[l_lbl], mu_qd[q_lbl])

        clipped = np.fmin(firing, _MFS_PHS[out_lbl])
        aggregated = np.fmax(aggregated, clipped)

        rule_log.append({
            "id":            f"R{idx}",
            "antecedents":   {"DSPD": d_lbl, "LTBF": l_lbl, "Qd": q_lbl},
            "consequent":    out_lbl,
            "firing_strength": float(firing),
            "rationale":     rationale,
        })


    if aggregated.sum() == 0:


        phs = 50.0
        dominant = "sustainable"
        dominant_mu = 0.0
    else:
        phs = float(fuzz.defuzz(PHS_UNIVERSE, aggregated, "centroid"))

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


def membership_curves() -> Dict[str, Dict]:
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
