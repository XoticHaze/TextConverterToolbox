from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_nonoverlap_phase_audit import BAR_NS, phases
from research.mnq_opportunity_target_matrix import model, quarter_starts

CONFIGS = {
    "h12_bar05": (12, 0.5),
    "h12_bar10": (12, 1.0),
    "h24_bar05": (24, 0.5),
    "h24_bar10": (24, 1.0),
}
COST_FLOOR = 0.0002
EXECUTION_COST = 0.0002


def utc_slot(ts: pd.Series) -> np.ndarray:
    return (ts.astype("int64").to_numpy() // BAR_NS).astype(np.int64)


def first_true(x: np.ndarray) -> int | None:
    idx = np.flatnonzero(x)
    return int(idx[0]) if len(idx) else None


def build_events(work: pd.DataFrame, horizon: int, mult: float, phase: int, features: list[str]) -> pd.DataFrame:
    rows = []
    slots = work["utc_slot"].to_numpy(np.int64)
    close = work["close"].to_numpy(float)
    high = work["high"].to_numpy(float)
    low = work["low"].to_numpy(float)
    rv = work["rv_120"].to_numpy(float)

    for i in range(len(work) - horizon):
        if slots[i] % horizon != phase:
            continue
        threshold = max(COST_FLOOR, float(mult * rv[i] * math.sqrt(horizon)))
        if not np.isfinite(threshold) or threshold <= 0:
            continue
        upper = close[i] * (1.0 + threshold)
        lower = close[i] * (1.0 - threshold)
        hi = high[i + 1:i + horizon + 1]
        lo = low[i + 1:i + horizon + 1]
        up_at = first_true(hi >= upper)
        down_at = first_true(lo <= lower)
        ambiguous = up_at is not None and down_at is not None and up_at == down_at

        if ambiguous:
            label = 0
            long_ret = -threshold
            short_ret = -threshold
        elif up_at is not None and (down_at is None or up_at < down_at):
            label = 1
            long_ret = threshold
            short_ret = -threshold
        elif down_at is not None and (up_at is None or down_at < up_at):
            label = -1
            long_ret = -threshold
            short_ret = threshold
        else:
            label = 0
            fwd = close[i + horizon] / close[i] - 1.0
            long_ret = float(fwd)
            short_ret = float(-fwd)

        row = {
            "timestamp": work["timestamp"].iloc[i],
            "event_end_timestamp": work["timestamp"].iloc[i + horizon],
            "target": int(label),
            "threshold": float(threshold),
            "long_return": float(long_ret),
            "short_return": float(short_ret),
            "ambiguous_same_bar": bool(ambiguous),
        }
        for f in features:
            row[f] = float(work[f].iloc[i])
        rows.append(row)
    return pd.DataFrame(rows)


def classification(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def realized_returns(events: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    directional = pred != 0
    if not directional.any():
        return np.asarray([], dtype=float)
    p = pred[directional]
    e = events.iloc[np.flatnonzero(directional)]
    gross = np.where(p == 1, e["long_return"].to_numpy(float), e["short_return"].to_numpy(float))
    return gross - EXECUTION_COST


def max_drawdown(arr: np.ndarray) -> float | None:
    if len(arr) == 0:
        return None
    curve = np.cumsum(arr)
    peak = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    return float(np.min(curve - peak))


def summarize(returns: list[float], q_net: list[float]) -> dict:
    r = np.asarray(returns, dtype=float)
    q = np.asarray(q_net, dtype=float)
    if len(r) == 0:
        return {"signals": 0}
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    t_stat = float(np.mean(r) / (std / math.sqrt(len(r)))) if std > 0 else None
    total = float(np.sum(r))
    quarter_abs = np.abs(q)
    concentration = float(np.max(quarter_abs) / np.sum(quarter_abs)) if len(q) and np.sum(quarter_abs) > 0 else None
    return {
        "signals": int(len(r)),
        "mean_net_after_2bp": float(np.mean(r)),
        "median_net_after_2bp": float(np.median(r)),
        "positive_rate": float(np.mean(r > 0)),
        "t_stat_zero_mean": t_stat,
        "cumulative_net_return_sum": total,
        "max_cumulative_drawdown": max_drawdown(r),
        "quarters": int(len(q)),
        "quarters_positive": int(np.sum(q > 0)),
        "median_quarter_net_after_2bp": float(np.median(q)) if len(q) else None,
        "min_quarter_net_after_2bp": float(np.min(q)) if len(q) else None,
        "max_quarter_net_after_2bp": float(np.max(q)) if len(q) else None,
        "largest_abs_quarter_share": concentration,
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
    cols = list(dict.fromkeys(["timestamp", "high", "low", "close", "rv_120", *expanded]))
    work = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    work["utc_slot"] = utc_slot(work["timestamp"])
    outer = quarter_starts()
    result_features = {}

    for feature_name, features in feature_sets.items():
        phase_nodes = {}
        for phase in phases(horizon):
            events = build_events(work, horizon, mult, phase, features)
            if len(events) < 1000:
                raise RuntimeError(f"{args.config_key}/{feature_name}/phase{phase}: too few events {len(events)}")
            all_returns: list[float] = []
            quarter_net: list[float] = []
            quarter_rows = []
            all_y: list[int] = []
            all_pred: list[int] = []

            for i in range(len(outer) - 1):
                start, end = outer[i], outer[i + 1]
                train = events[(events["timestamp"] < start) & (events["event_end_timestamp"] < start)]
                test = events[(events["timestamp"] >= start) & (events["timestamp"] < end) & (events["event_end_timestamp"] < end)]
                if len(train) < 500 or len(test) < 30:
                    continue
                y_train = train["target"].astype(int).to_numpy()
                y_test = test["target"].astype(int).to_numpy()
                # Tight barriers can legitimately collapse to a binary up/down target.
                # Only a single observed training class is unusable.
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                fitted = model().fit(train[features].to_numpy(float), y_train)
                pred = fitted.predict(test[features].to_numpy(float)).astype(int)
                r = realized_returns(test, pred)
                qmean = float(np.mean(r)) if len(r) else None
                if len(r):
                    all_returns.extend(r.tolist())
                    quarter_net.append(qmean)
                all_y.extend(y_test.tolist())
                all_pred.extend(pred.tolist())
                quarter_rows.append({
                    "period": f"{start.year}Q{((start.month - 1)//3)+1}",
                    "train_events": int(len(train)),
                    "test_events": int(len(test)),
                    "directional_signals": int(len(r)),
                    "classification": classification(y_test, pred),
                    "mean_net_after_2bp": qmean,
                    "target_counts": {str(c): int(np.sum(y_test == c)) for c in (-1, 0, 1)},
                    "observed_train_classes": sorted(int(c) for c in np.unique(y_train)),
                    "ambiguous_events": int(test["ambiguous_same_bar"].sum()),
                })

            if len(quarter_rows) < 8:
                raise RuntimeError(f"{args.config_key}/{feature_name}/phase{phase}: insufficient outer quarters {len(quarter_rows)}")
            phase_nodes[str(phase)] = {
                "events_total": int(len(events)),
                "quarter_rows": quarter_rows,
                "aggregate_classification": classification(np.asarray(all_y, dtype=int), np.asarray(all_pred, dtype=int)),
                "aggregate_economic": summarize(all_returns, quarter_net),
            }
            print(args.config_key, feature_name, "PHASE", phase, phase_nodes[str(phase)]["aggregate_economic"])

        result_features[feature_name] = {"phases": phase_nodes}

    result = {
        "schema": "foundry.mnq_triple_barrier_events.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon": horizon,
        "barrier_multiplier": mult,
        "protocol": "four predeclared absolute-UTC phase streams; event starts separated by full horizon; phase-specific models train only on prior completed non-overlapping events; symmetric causal rv_120 barriers with 2bp floor; binary or ternary observed targets are valid; first barrier wins; same-bar dual hit is fail-closed as directional loss; expiry exits at horizon close; current quarter never used for model selection",
        "execution_cost_per_event": EXECUTION_COST,
        "excluded_forward_aligned_features": ["chikou_span"],
        "feature_sets": result_features,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_TRIPLE_BARRIER_EVENTS=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
