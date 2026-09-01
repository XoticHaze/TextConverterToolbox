from __future__ import annotations

"""Bounded dependency subset for the model-family research branch.

This branch only needs the already-used MNQ point/non-overlap phase contract from the
NQ->MNQ transfer experiment. No transfer-model CLI or promotion authority is defined here.
"""

import numpy as np
import pandas as pd

BAR = pd.Timedelta(minutes=12)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
MNQ_DOLLARS_PER_POINT = 2.0
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)


def utc_slot(ts: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    return ((parsed - EPOCH) // BAR).to_numpy(dtype=np.int64)


def signed_points(pred: np.ndarray, point_move: np.ndarray) -> np.ndarray:
    selected = pred != 0
    if not selected.any():
        return np.asarray([], dtype=float)
    side = np.where(pred[selected] == 1, 1.0, -1.0)
    return side * point_move[selected]


def phase_audit(ts: pd.Series, pred: np.ndarray, point_move: np.ndarray, horizon: int) -> dict:
    slots = utc_slot(ts)
    phases: dict[str, dict] = {}
    for phase in range(horizon):
        mask = (slots % horizon) == phase
        gross = signed_points(pred[mask], point_move[mask])
        rec = {"signals": int(len(gross))}
        if len(gross) >= 10:
            rec["gross_mean_points"] = float(np.mean(gross))
            for cost in POINT_COSTS:
                key = str(cost).replace(".", "p")
                mean_points = float(np.mean(gross - cost))
                rec[f"net_mean_points_after_{key}pt"] = mean_points
                rec[f"net_mean_mnq_dollars_after_{key}pt"] = mean_points * MNQ_DOLLARS_PER_POINT
        phases[str(phase)] = rec
    valid = [v for v in phases.values() if "net_mean_points_after_1p0pt" in v]

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(v[key]) for v in valid], dtype=float)

    out = {
        "phase_streams": phases,
        "valid_phases": int(len(valid)),
        "contract": f"absolute UTC 12-minute slot modulo H{horizon}; within-phase forecasts separated by at least {horizon * 12} minutes; every phase reported, none selected post-hoc",
    }
    if valid:
        for cost in POINT_COSTS:
            key = str(cost).replace(".", "p")
            vals = arr(f"net_mean_points_after_{key}pt")
            out[f"median_phase_net_points_after_{key}pt"] = float(np.median(vals))
            out[f"mean_phase_net_points_after_{key}pt"] = float(np.mean(vals))
            out[f"positive_phase_fraction_after_{key}pt"] = float(np.mean(vals > 0))
            out[f"median_phase_net_mnq_dollars_after_{key}pt"] = float(np.median(vals) * MNQ_DOLLARS_PER_POINT)
    return out
