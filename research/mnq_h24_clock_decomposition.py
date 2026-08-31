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
from research.mnq_nonoverlap_phase_audit import BAR_NS
from research.mnq_opportunity_target_matrix import model, quarter_starts, target_columns

CONFIGS = {
    "h24_vol05": (24, 0.5),
    "h24_vol10": (24, 1.0),
}
PHASES = (12, 18)
COST = 0.0002


def utc_slot(ts: pd.Series) -> np.ndarray:
    # Force nanosecond resolution before integer conversion. Pandas may preserve
    # source datetime64[us] resolution; dividing microseconds by BAR_NS silently
    # destroys the absolute 12-minute phase contract.
    ns = pd.to_datetime(ts, utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return (ns // BAR_NS).astype(np.int64)


def signed_net(pred: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    mask = pred != 0
    if not mask.any():
        return np.asarray([], dtype=float)
    direction = np.where(pred[mask] == 1, 1.0, -1.0)
    return direction * fwd[mask] - COST


def summarize(returns: list[float], quarter_means: list[float]) -> dict:
    r = np.asarray(returns, dtype=float)
    q = np.asarray(quarter_means, dtype=float)
    if len(r) == 0:
        return {"signals": 0}
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    t_stat = float(np.mean(r) / (std / math.sqrt(len(r)))) if std > 0 else None
    curve = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    return {
        "signals": int(len(r)),
        "mean_net_after_2bp": float(np.mean(r)),
        "median_net_after_2bp": float(np.median(r)),
        "positive_rate": float(np.mean(r > 0)),
        "t_stat_zero_mean": t_stat,
        "max_cumulative_drawdown": float(np.min(curve - peak)),
        "quarters_observed": int(len(q)),
        "quarters_positive": int(np.sum(q > 0)),
        "median_quarter_net_after_2bp": float(np.median(q)) if len(q) else None,
        "min_quarter_net_after_2bp": float(np.min(q)) if len(q) else None,
        "max_quarter_net_after_2bp": float(np.max(q)) if len(q) else None,
    }


def clock_label(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}Z"


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
    work["utc_minute"] = (work["timestamp"].dt.hour * 60 + work["timestamp"].dt.minute).astype(int)
    outer = quarter_starts()

    result_features = {}
    for feature_name, features in feature_sets.items():
        clock_returns: dict[int, list[float]] = {}
        clock_quarters: dict[int, list[float]] = {}
        quarter_rows = []

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
            minutes = test["utc_minute"].to_numpy(int)

            q = {"period": f"{start.year}Q{((start.month - 1)//3)+1}", "clocks": {}}
            for phase in PHASES:
                phase_mask = (slots % horizon) == phase
                for minute in sorted(set(minutes[phase_mask].tolist())):
                    cmask = phase_mask & (minutes == minute)
                    r = signed_net(pred[cmask], fwd_test[cmask])
                    clock_returns.setdefault(minute, []).extend(r.tolist())
                    if len(r):
                        qmean = float(np.mean(r))
                        clock_quarters.setdefault(minute, []).append(qmean)
                    else:
                        qmean = None
                    q["clocks"][clock_label(minute)] = {
                        "phase": phase,
                        "signals": int(len(r)),
                        "mean_net_after_2bp": qmean,
                    }
            quarter_rows.append(q)

        expected_minutes = sorted(clock_returns)
        if len(expected_minutes) != 10:
            raise RuntimeError(f"expected 10 H24 phase clock buckets, got {[(m, clock_label(m)) for m in expected_minutes]}")
        clocks = {}
        for minute in expected_minutes:
            phase = int((minute // 12) % horizon)
            clocks[clock_label(minute)] = {
                "phase": phase,
                "aggregate": summarize(clock_returns[minute], clock_quarters.get(minute, [])),
            }
        result_features[feature_name] = {"clocks": clocks, "quarter_rows": quarter_rows}

    result = {
        "schema": "foundry.mnq_h24_clock_decomposition.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": mult,
        "phases_explained": list(PHASES),
        "protocol": "explanatory-only decomposition of the two H24 phases identified by the completed non-overlap audit; all ten fixed UTC clock buckets reported with no bucket selection; quarterly expanding past-only model refits; each exact clock bucket occurs once per day and is non-overlapping at H24; 2bp sensitivity",
        "excluded_forward_aligned_features": ["chikou_span"],
        "feature_sets": result_features,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_CLOCK_DECOMPOSITION=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    for feature_name, node in result_features.items():
        print(feature_name)
        for clock, metrics in node["clocks"].items():
            print(clock, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
