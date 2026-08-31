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
from research.mnq_nonoverlap_phase_audit import BAR_NS
from research.mnq_opportunity_target_matrix import model, quarter_starts

CONFIGS = {
    "h6_k10": (6, 1.0),
    "h12_k075": (12, 0.75),
    "h12_k10": (12, 1.0),
    "h24_k10": (24, 1.0),
}
COST_FLOOR = 0.0002


def utc_slot(ts: pd.Series) -> np.ndarray:
    return (ts.astype("int64").to_numpy() // BAR_NS).astype(np.int64)


def build_events(frame: pd.DataFrame, horizon: int, mult: float) -> tuple[pd.DataFrame, dict]:
    """Create absolute-clock, non-overlapping events with causal barriers.

    Events start only when absolute UTC 12-minute slot % horizon == 0. Their
    vertical barrier is exactly horizon * 12 minutes later, so event windows do
    not overlap in clock time. Barrier width is fixed at entry from causal
    rv_120. If both barriers are first touched inside the same future bar the
    event is ambiguous and excluded rather than assuming intrabar ordering.
    """
    work = frame.copy().reset_index(drop=True)
    work["utc_slot"] = utc_slot(work["timestamp"])
    starts = np.flatnonzero((work["utc_slot"].to_numpy(np.int64) % horizon) == 0)
    rows = []
    ambiguous = 0
    no_future = 0
    for i in starts:
        entry = work.iloc[i]
        if not np.isfinite(float(entry["rv_120"])):
            continue
        entry_ts = entry["timestamp"]
        vertical_ts = entry_ts + pd.Timedelta(minutes=12 * horizon)
        future = work[(work["timestamp"] > entry_ts) & (work["timestamp"] <= vertical_ts)]
        if future.empty:
            no_future += 1
            continue
        threshold = max(COST_FLOOR, mult * float(entry["rv_120"]) * math.sqrt(horizon))
        entry_close = float(entry["close"])
        upper = entry_close * math.exp(threshold)
        lower = entry_close * math.exp(-threshold)
        label = 0
        exit_kind = "vertical"
        exit_ts = future["timestamp"].iloc[-1]
        realized = float(future["close"].iloc[-1] / entry_close - 1.0)
        bad = False
        for r in future.itertuples(index=False):
            up = float(r.high) >= upper
            dn = float(r.low) <= lower
            if up and dn:
                ambiguous += 1
                bad = True
                break
            if up:
                label = 1
                exit_kind = "upper"
                exit_ts = r.timestamp
                realized = math.exp(threshold) - 1.0
                break
            if dn:
                label = -1
                exit_kind = "lower"
                exit_ts = r.timestamp
                realized = math.exp(-threshold) - 1.0
                break
        if bad:
            continue
        node = {
            "timestamp": entry_ts,
            "vertical_timestamp": vertical_ts,
            "exit_timestamp": exit_ts,
            "label": label,
            "threshold_log": threshold,
            "realized_event_return": realized,
            "exit_kind": exit_kind,
        }
        for c in BASE_FEATURES:
            node[c] = float(entry[c])
        for c in EXPANDED_FEATURES:
            node[c] = float(entry[c])
        for c in REGIME_FEATURES:
            node[c] = float(entry[c])
        rows.append(node)
    events = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if events.empty:
        raise RuntimeError("no triple-barrier events")
    # Windows must be non-overlapping by construction.
    ts = events["timestamp"].astype("int64").to_numpy()
    if len(ts) > 1 and np.any(np.diff(ts) < horizon * BAR_NS):
        raise RuntimeError("event clock windows overlap")
    return events, {
        "events": int(len(events)),
        "ambiguous_same_bar_first_touch_excluded": int(ambiguous),
        "event_starts_without_future_bars": int(no_future),
        "label_counts": {str(c): int((events["label"] == c).sum()) for c in (-1, 0, 1)},
    }


def classification(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def signal_returns(pred: np.ndarray, label: np.ndarray, threshold: np.ndarray, vertical_ret: np.ndarray) -> np.ndarray:
    selected = pred != 0
    if not selected.any():
        return np.asarray([], dtype=float)
    p = pred[selected]
    y = label[selected]
    th = threshold[selected]
    vr = vertical_ret[selected]
    out = np.empty(len(p), dtype=float)
    for i, (side, truth, barrier, vert) in enumerate(zip(p, y, th, vr)):
        if truth == 0:
            out[i] = float(side) * float(vert)
        elif side == truth:
            out[i] = math.exp(float(barrier)) - 1.0
        else:
            out[i] = math.exp(-float(barrier)) - 1.0
    return out


def economics(pred: np.ndarray, label: np.ndarray, threshold: np.ndarray, vertical_ret: np.ndarray) -> dict | None:
    r = signal_returns(pred, label, threshold, vertical_ret)
    if len(r) < 30:
        return None
    curve = np.cumsum(r - 0.0002)
    peak = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    std = float(np.std(r - 0.0002, ddof=1)) if len(r) > 1 else 0.0
    tstat = float(np.mean(r - 0.0002) / (std / math.sqrt(len(r)))) if std > 0 else None
    selected = pred != 0
    return {
        "signals": int(len(r)),
        "coverage": float(selected.mean()),
        "long_signals": int((pred[selected] == 1).sum()),
        "short_signals": int((pred[selected] == -1).sum()),
        "gross_mean_event_return": float(np.mean(r)),
        "gross_median_event_return": float(np.median(r)),
        "positive_rate": float(np.mean(r > 0)),
        "net_mean_after_2bp": float(np.mean(r - 0.0002)),
        "net_mean_after_5bp": float(np.mean(r - 0.0005)),
        "net_mean_after_10bp": float(np.mean(r - 0.0010)),
        "t_stat_net_2bp_zero_mean": tstat,
        "max_cumulative_drawdown_after_2bp": float(np.min(curve - peak)),
        "note": "research event economics using barrier threshold fills; ambiguous same-bar first touches excluded; costs are sensitivity, not execution proof",
    }


def summarize(rows: list[dict], all_pred: list[int], all_label: list[int], all_th: list[float], all_ret: list[float]) -> dict:
    p = np.asarray(all_pred, dtype=int)
    y = np.asarray(all_label, dtype=int)
    th = np.asarray(all_th, dtype=float)
    vr = np.asarray(all_ret, dtype=float)
    qnet = [r["economic"]["net_mean_after_2bp"] for r in rows if r["economic"]]
    return {
        "quarters": len(rows),
        "quarters_positive_after_2bp": int(sum(v > 0 for v in qnet)),
        "median_quarter_net_after_2bp": float(np.median(qnet)) if qnet else None,
        "min_quarter_net_after_2bp": float(np.min(qnet)) if qnet else None,
        "max_quarter_net_after_2bp": float(np.max(qnet)) if qnet else None,
        "classification": classification(y, p),
        "economic": economics(p, y, th, vr),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, mult = CONFIGS[args.config_key]

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    needed = list(dict.fromkeys(["timestamp", "open", "high", "low", "close", "rv_120", *expanded]))
    frame = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    events, event_meta = build_events(frame, horizon, mult)

    feature_sets = {"baseline20": list(BASE_FEATURES), "expanded_regime": expanded}
    outer = quarter_starts()
    results = {}
    for fname, features in feature_sets.items():
        rows = []
        all_pred: list[int] = []
        all_label: list[int] = []
        all_th: list[float] = []
        all_ret: list[float] = []
        for i in range(len(outer) - 1):
            start, end = outer[i], outer[i + 1]
            train = events[events["vertical_timestamp"] < start]
            test = events[(events["timestamp"] >= start) & (events["timestamp"] < end)]
            if len(train) < 1000 or len(test) < 100:
                continue
            ytr = train["label"].astype(int).to_numpy()
            yte = test["label"].astype(int).to_numpy()
            if len(np.unique(ytr)) < 3 or len(np.unique(yte)) < 3:
                continue
            fitted = model().fit(train[features].to_numpy(float), ytr)
            pred = fitted.predict(test[features].to_numpy(float)).astype(int)
            th = test["threshold_log"].to_numpy(float)
            vr = test["realized_event_return"].to_numpy(float)
            econ = economics(pred, yte, th, vr)
            row = {
                "period": f"{start.year}Q{((start.month - 1)//3)+1}",
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "train_max_vertical_timestamp": train["vertical_timestamp"].max().isoformat(),
                "test_first_timestamp": test["timestamp"].min().isoformat(),
                "target_counts": {str(c): int((yte == c).sum()) for c in (-1, 0, 1)},
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "classification": classification(yte, pred),
                "economic": econ,
            }
            rows.append(row)
            all_pred.extend(pred.tolist()); all_label.extend(yte.tolist()); all_th.extend(th.tolist()); all_ret.extend(vr.tolist())
            print(args.config_key, fname, row["period"], "BA", row["classification"]["balanced_accuracy"], "NET2", None if econ is None else econ["net_mean_after_2bp"], "SIGNALS", None if econ is None else econ["signals"])
        if len(rows) < 8:
            raise RuntimeError(f"{args.config_key}/{fname}: insufficient prospective quarters {len(rows)}")
        results[fname] = {"quarter_rows": rows, "aggregate": summarize(rows, all_pred, all_label, all_th, all_ret)}

    result = {
        "schema": "foundry.mnq_triple_barrier_events.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon_bars": horizon,
        "barrier_multiplier": mult,
        "barrier": "symmetric log-price barrier max(2bp, multiplier * causal rv_120 * sqrt(horizon)) fixed at event entry",
        "event_schedule": "absolute UTC 12-minute slot modulo horizon == 0; vertical barrier exactly horizon*12 minutes; non-overlapping clock windows",
        "label": "first upper hit=+1, first lower hit=-1, vertical expiry=0; same-bar dual first hit excluded as ambiguous",
        "protocol": "quarterly expanding past-only event refit; training requires event vertical timestamp strictly before outer-quarter start; fixed outer quarters; no threshold selection from future outcomes",
        "excluded_forward_aligned_features": ["chikou_span"],
        "event_meta": event_meta,
        "feature_sets": results,
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
