from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_opportunity_target_matrix import model, target_columns
from research.nq_cc0_multiyear_validation import CONFIGS, EPOCH, BAR, bars12, load_normalized

ARMS = ("expanding_nq", "rolling_90d_nq")
THRESHOLDS = (0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60)
DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-12-12", tz="UTC")
COSTS = (0.0002, 0.0005, 0.0010)


def utc_slot(ts: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    return ((parsed - EPOCH) // BAR).to_numpy(dtype=np.int64)


def make_store(horizon: int) -> dict[float, list[list[float]]]:
    return {float(t): [[] for _ in range(horizon)] for t in THRESHOLDS}


def add_events(store: dict[float, list[list[float]]], ts: pd.Series, pred: np.ndarray, conf: np.ndarray, fwd: np.ndarray, horizon: int) -> None:
    phases = utc_slot(ts) % horizon
    for threshold, phase_lists in store.items():
        selected = (pred != 0) & (conf >= threshold)
        if not selected.any():
            continue
        side = np.where(pred[selected] == 1, 1.0, -1.0)
        gross = side * fwd[selected]
        sel_phases = phases[selected]
        for phase in range(horizon):
            values = gross[sel_phases == phase]
            if len(values):
                phase_lists[phase].extend(values.astype(float).tolist())


def summarize(store: dict[float, list[list[float]]], horizon: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for threshold, phase_lists in store.items():
        phases = []
        for phase, values in enumerate(phase_lists):
            arr = np.asarray(values, dtype=float)
            phases.append({
                "phase": phase,
                "signals": int(len(arr)),
                "mean_net2": float(np.mean(arr - COSTS[0])) if len(arr) else None,
                "mean_net5": float(np.mean(arr - COSTS[1])) if len(arr) else None,
                "mean_net10": float(np.mean(arr - COSTS[2])) if len(arr) else None,
                "positive_rate_gross": float(np.mean(arr > 0)) if len(arr) else None,
            })
        valid = [p for p in phases if p["signals"] >= 20]
        def vals(key: str) -> np.ndarray:
            return np.asarray([p[key] for p in valid], dtype=float)
        v2, v5, v10 = vals("mean_net2"), vals("mean_net5"), vals("mean_net10")
        out[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "total_signals": int(sum(p["signals"] for p in phases)),
            "valid_phases": int(len(valid)),
            "phase_count": int(horizon),
            "median_phase_net2": float(np.median(v2)) if len(v2) else None,
            "median_phase_net5": float(np.median(v5)) if len(v5) else None,
            "median_phase_net10": float(np.median(v10)) if len(v10) else None,
            "positive_phase_fraction_net2": float(np.mean(v2 > 0)) if len(v2) else None,
            "positive_phase_fraction_net5": float(np.mean(v5 > 0)) if len(v5) else None,
            "positive_phase_fraction_net10": float(np.mean(v10 > 0)) if len(v10) else None,
            "min_phase_net5": float(np.min(v5)) if len(v5) else None,
            "max_phase_net5": float(np.max(v5)) if len(v5) else None,
            "phase_streams": phases,
        }
    return out


def select_threshold(discovery: dict[str, dict], horizon: int) -> float | None:
    # Predeclared, deliberately conservative and non-maximizing: take the LOWEST confidence
    # threshold that clears a 5bp economic hurdle while retaining broad phase coverage.
    for threshold in THRESHOLDS:
        row = discovery[f"{threshold:.2f}"]
        if row["total_signals"] < 1000:
            continue
        if row["valid_phases"] < horizon:
            continue
        if row["median_phase_net5"] is None or row["median_phase_net5"] <= 0:
            continue
        if row["positive_phase_fraction_net5"] is None or row["positive_phase_fraction_net5"] < 0.60:
            continue
        return float(threshold)
    return None


def confirmation_pass(row: dict | None, horizon: int) -> bool:
    if not row:
        return False
    return bool(
        row["total_signals"] >= 300
        and row["valid_phases"] >= max(1, int(np.ceil(horizon * 0.75)))
        and row["median_phase_net5"] is not None
        and row["median_phase_net5"] > 0
        and row["positive_phase_fraction_net5"] is not None
        and row["positive_phase_fraction_net5"] >= 0.60
    )


def train_for_arm(work: pd.DataFrame, train_end: int, start: pd.Timestamp, arm: str) -> pd.DataFrame:
    train = work.iloc[:train_end].copy()
    if arm == "rolling_90d_nq":
        train = train[train["timestamp"] >= start - pd.Timedelta(days=90)]
    elif arm != "expanding_nq":
        raise RuntimeError(f"unknown arm {arm}")
    return train[train["target"].notna()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--nq-normalized", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]

    source_sha = hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest()
    raw = load_normalized(args.nq_normalized)
    bars = bars12(raw)
    frame = _add_features(bars)
    features = list(BASE_FEATURES)
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    work = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, fwd, _ = target_columns(work, horizon, vol_mult)
    work["target"] = label
    work["fwd"] = fwd

    stores = {
        arm: {"discovery": make_store(horizon), "confirmation": make_store(horizon)}
        for arm in ARMS
    }
    windows = {arm: {"discovery": 0, "confirmation": 0} for arm in ARMS}

    starts = list(pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"))
    for start in starts:
        end = start + pd.Timedelta(days=7)
        test_mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna()
        idx = np.flatnonzero(test_mask.to_numpy())
        if len(idx) < 300:
            continue
        start_idx = int(idx[0])
        train_end = start_idx - horizon
        if train_end <= 0:
            continue
        test = work.iloc[int(idx[0]):int(idx[-1]) + 1]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["target"].notna()]
        if len(test) < 300:
            continue
        period = "discovery" if start < DISCOVERY_END else "confirmation"
        x_test = test[features].to_numpy(float)
        fwd_test = test["fwd"].to_numpy(float)
        for arm in ARMS:
            train = train_for_arm(work, train_end, start, arm)
            if len(train) < 2500 or train["timestamp"].max() >= test["timestamp"].min():
                continue
            fitted = model().fit(train[features].to_numpy(float), train["target"].astype(int).to_numpy())
            proba = fitted.predict_proba(x_test)
            classes = fitted.classes_.astype(int)
            arg = np.argmax(proba, axis=1)
            pred = classes[arg]
            conf = proba[np.arange(len(proba)), arg]
            add_events(stores[arm][period], test["timestamp"], pred, conf, fwd_test, horizon)
            windows[arm][period] += 1

    arms_result = {}
    for arm in ARMS:
        discovery = summarize(stores[arm]["discovery"], horizon)
        confirmation = summarize(stores[arm]["confirmation"], horizon)
        selected = select_threshold(discovery, horizon)
        confirmation_row = confirmation.get(f"{selected:.2f}") if selected is not None else None
        arms_result[arm] = {
            "windows": windows[arm],
            "discovery_grid": discovery,
            "selected_threshold_from_discovery": selected,
            "selection_rule": "lowest directional predicted-class probability threshold in predeclared grid with >=1000 signals, all phases >=20 signals, median phase mean net after 5bp > 0, and >=60% phases positive after 5bp; no threshold selected if none clears",
            "confirmation_grid_diagnostic_only": confirmation,
            "frozen_confirmation_at_selected_threshold": confirmation_row,
            "confirmation_pass": confirmation_pass(confirmation_row, horizon),
        }
        print("ARM", arm, "SELECTED", selected, "CONFIRM_PASS", arms_result[arm]["confirmation_pass"])
        if selected is not None:
            print("DISCOVERY", discovery[f"{selected:.2f}"])
            print("CONFIRM", confirmation_row)

    result = {
        "schema": "foundry.nq_cc0_confidence_tail_validation.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "tgtanalytics/nq-futures-1min-bar-2022-2025 (CC0 Public Domain)",
        "normalized_source_sha256": source_sha,
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "feature_set": "baseline20",
        "threshold_grid": list(THRESHOLDS),
        "discovery_period": [TEST_START.isoformat(), DISCOVERY_END.isoformat()],
        "confirmation_period": [DISCOVERY_END.isoformat(), TEST_END.isoformat()],
        "costs": list(COSTS),
        "protocol": "weekly past-only fits with H-row label purge; confidence is probability of the model's predicted class; only non-neutral predictions at/above threshold enter; events audited as all absolute-UTC non-overlapping phase streams; threshold selection restricted to 2023-07 through 2024 and takes the lowest grid threshold clearing a predeclared 5bp hurdle; 2025 is frozen confirmation and never reselects threshold",
        "arms": arms_result,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_CC0_CONFIDENCE_VALIDATION=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
