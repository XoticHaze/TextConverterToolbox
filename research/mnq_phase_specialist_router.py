from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_nonoverlap_phase_audit import BAR_NS, phases
from research.mnq_opportunity_target_matrix import model, quarter_starts, target_columns

CONFIGS = {
    "h12_vol10": (12, 1.0),
    "h24_vol05": (24, 0.5),
    "h24_vol10": (24, 1.0),
}
LOOKBACK_QUARTERS = 4
COST = 0.0002


def utc_slot(ts: pd.Series) -> np.ndarray:
    return (ts.astype("int64").to_numpy() // BAR_NS).astype(np.int64)


def event_returns(pred: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    mask = pred != 0
    if not mask.any():
        return np.asarray([], dtype=float)
    direction = np.where(pred[mask] == 1, 1.0, -1.0)
    return direction * fwd[mask] - COST


def max_drawdown(returns: np.ndarray) -> float | None:
    if len(returns) == 0:
        return None
    curve = np.cumsum(returns)
    peak = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    return float(np.min(curve - peak))


def aggregate(returns: list[float], quarter_net: list[float]) -> dict:
    arr = np.asarray(returns, dtype=float)
    q = np.asarray(quarter_net, dtype=float)
    if len(arr) == 0:
        return {"signals": 0}
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    t_stat = float(np.mean(arr) / (std / math.sqrt(len(arr)))) if std > 0 else None
    return {
        "signals": int(len(arr)),
        "mean_net_after_2bp": float(np.mean(arr)),
        "median_net_after_2bp": float(np.median(arr)),
        "positive_rate": float(np.mean(arr > 0)),
        "t_stat_zero_mean": t_stat,
        "max_cumulative_drawdown": max_drawdown(arr),
        "quarters_traded": int(len(q)),
        "quarters_positive": int(np.sum(q > 0)),
        "median_quarter_net_after_2bp": float(np.median(q)) if len(q) else None,
        "min_quarter_net_after_2bp": float(np.min(q)) if len(q) else None,
        "max_quarter_net_after_2bp": float(np.max(q)) if len(q) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    horizon, mult = CONFIGS[args.config_key]
    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    frame = _add_features(deep_bars(stitched))
    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    feature_sets = {"baseline20": list(BASE_FEATURES), "expanded_regime": expanded}
    cols = list(dict.fromkeys(["timestamp", "close", "rv_120", *expanded]))
    work = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, fwd, _ = target_columns(work, horizon, mult)
    work["target"] = label
    work["fwd"] = fwd
    work["utc_slot"] = utc_slot(work["timestamp"])
    outer = quarter_starts()
    phase_ids = phases(horizon)
    results = {}

    for feature_name, features in feature_sets.items():
        history = {p: [] for p in phase_ids}
        router_rows = []
        selected_returns: list[float] = []
        selected_quarter_net: list[float] = []

        for i in range(len(outer) - 1):
            start, end = outer[i], outer[i + 1]
            mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna()
            idx = np.flatnonzero(mask.to_numpy())
            if len(idx) < 500:
                continue
            test_start, test_end = int(idx[0]), int(idx[-1] + 1)
            train_end = test_start - horizon
            if train_end < 10000:
                continue
            train = work.iloc[:train_end]
            train = train[train["target"].notna()]
            test = work.iloc[test_start:test_end]
            test = test[test["target"].notna()].copy()
            if len(train) < 10000 or len(test) < 500:
                continue
            y_train = train["target"].astype(int).to_numpy()
            if len(np.unique(y_train)) < 3:
                continue
            fitted = model().fit(train[features].to_numpy(float), y_train)
            pred = fitted.predict(test[features].to_numpy(float)).astype(int)
            fwd_test = test["fwd"].to_numpy(float)
            slots = test["utc_slot"].to_numpy(np.int64)

            phase_scores = {}
            for p in phase_ids:
                h = history[p][-LOOKBACK_QUARTERS:]
                if len(h) >= LOOKBACK_QUARTERS:
                    median = float(np.median(h))
                    positive_rate = float(np.mean(np.asarray(h) > 0))
                    phase_scores[p] = {"median_net2": median, "positive_rate": positive_rate}

            eligible = [p for p, s in phase_scores.items() if s["median_net2"] > 0 and s["positive_rate"] >= 0.5]
            chosen = max(eligible, key=lambda p: (phase_scores[p]["median_net2"], -p)) if eligible else None
            chosen_returns = np.asarray([], dtype=float)
            if chosen is not None:
                pmask = (slots % horizon) == chosen
                chosen_returns = event_returns(pred[pmask], fwd_test[pmask])
                if len(chosen_returns):
                    selected_returns.extend(chosen_returns.tolist())
                    selected_quarter_net.append(float(np.mean(chosen_returns)))

            current_phase_net = {}
            for p in phase_ids:
                pmask = (slots % horizon) == p
                r = event_returns(pred[pmask], fwd_test[pmask])
                net = float(np.mean(r)) if len(r) else None
                current_phase_net[p] = net

            router_rows.append({
                "period": f"{start.year}Q{((start.month - 1)//3)+1}",
                "chosen_phase": chosen,
                "selection_scores_from_prior_only": phase_scores,
                "chosen_signals": int(len(chosen_returns)),
                "chosen_current_quarter_net2": float(np.mean(chosen_returns)) if len(chosen_returns) else None,
                "current_phase_net2_for_future_selection_only": {str(k): v for k, v in current_phase_net.items()},
            })

            # Current quarter is appended only after the current decision/evaluation.
            for p, net in current_phase_net.items():
                if net is not None:
                    history[p].append(net)

        results[feature_name] = {
            "router_rows": router_rows,
            "aggregate": aggregate(selected_returns, selected_quarter_net),
        }

    result = {
        "schema": "foundry.mnq_phase_specialist_router.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": mult,
        "protocol": "quarterly expanding model refit; four fixed non-overlap timing specialists; exactly one phase may be routed per quarter; router uses only trailing four PRIOR outer-quarter phase net returns after 2bp; positive median and >=50% positive prior quarters required; no current-quarter selection leakage",
        "selection_lookback_quarters": LOOKBACK_QUARTERS,
        "cost_per_event": COST,
        "phase_offsets": phase_ids,
        "excluded_forward_aligned_features": ["chikou_span"],
        "feature_sets": results,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_PHASE_SPECIALIST_ROUTER=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    for name, node in results.items():
        print(name, node["aggregate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
