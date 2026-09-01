from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_opportunity_target_matrix import classification, model, target_columns

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
ARMS = ("expanding_nq", "rolling_90d_nq")
BAR = pd.Timedelta(minutes=12)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
COSTS = (0.0002, 0.0005, 0.0010)
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-12-12", tz="UTC")


def load_normalized(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, compression="infer")
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in f.columns]
    if missing:
        raise RuntimeError(f"missing normalized columns {missing}")
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="raise")
    for c in required[1:]:
        f[c] = pd.to_numeric(f[c], errors="raise")
    f = f.sort_values("timestamp").reset_index(drop=True)
    if f["timestamp"].duplicated().any() or not f["timestamp"].is_monotonic_increasing:
        raise RuntimeError("normalized NQ timestamps not unique/increasing")
    return f[required]


def bars12(frame: pd.DataFrame) -> pd.DataFrame:
    b = frame.set_index("timestamp").resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), observed_minutes=("close", "count"),
    )
    b = b[b["observed_minutes"] > 0].reset_index()
    b["market"] = "NQ"
    return b


def utc_slot(ts: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    return ((parsed - EPOCH) // BAR).to_numpy(dtype=np.int64)


def signed_returns(pred: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    selected = pred != 0
    if not selected.any():
        return np.asarray([], dtype=float)
    side = np.where(pred[selected] == 1, 1.0, -1.0)
    return side * fwd[selected]


def phase_audit(ts: pd.Series, pred: np.ndarray, fwd: np.ndarray, horizon: int) -> dict:
    slots = utc_slot(ts)
    phases = {}
    for phase in range(horizon):
        mask = (slots % horizon) == phase
        gross = signed_returns(pred[mask], fwd[mask])
        phases[str(phase)] = {
            "signals": int(len(gross)),
            "mean_net_after_2bp": float(np.mean(gross - 0.0002)) if len(gross) >= 10 else None,
            "mean_net_after_5bp": float(np.mean(gross - 0.0005)) if len(gross) >= 10 else None,
            "mean_net_after_10bp": float(np.mean(gross - 0.0010)) if len(gross) >= 10 else None,
            "positive_rate_gross": float(np.mean(gross > 0)) if len(gross) >= 10 else None,
        }
    valid = [v for v in phases.values() if v["mean_net_after_2bp"] is not None]
    def vals(key: str) -> list[float]:
        return [float(v[key]) for v in valid]
    net2 = vals("mean_net_after_2bp")
    return {
        "phase_streams": phases,
        "valid_phases": int(len(valid)),
        "positive_phases_net2": int(sum(v > 0 for v in net2)),
        "positive_phase_fraction_net2": float(np.mean(np.asarray(net2) > 0)) if net2 else None,
        "median_phase_net_after_2bp": float(np.median(net2)) if net2 else None,
        "median_phase_net_after_5bp": float(np.median(vals("mean_net_after_5bp"))) if valid else None,
        "median_phase_net_after_10bp": float(np.median(vals("mean_net_after_10bp"))) if valid else None,
        "contract": f"absolute UTC 12-minute slot modulo H{horizon}; signals within each phase separated by at least {horizon * 12} clock minutes; all phases reported and none selected post-hoc",
    }


def arm_train(work: pd.DataFrame, start_idx: int, train_end: int, start: pd.Timestamp, arm: str) -> pd.DataFrame:
    train = work.iloc[:train_end].copy()
    if arm == "rolling_90d_nq":
        train = train[train["timestamp"] >= start - pd.Timedelta(days=90)]
    elif arm != "expanding_nq":
        raise RuntimeError(f"unknown arm {arm}")
    return train[train["target"].notna()]


def summarize(rows: list[dict], arm: str) -> dict:
    med2 = [r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_2bp"] for r in rows]
    med5 = [r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_5bp"] for r in rows]
    med10 = [r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_10bp"] for r in rows]
    frac = [r["arms"][arm]["nonoverlap_phase_audit"]["positive_phase_fraction_net2"] for r in rows]
    valid = [(a, b, c, d) for a, b, c, d in zip(med2, med5, med10, frac) if a is not None and b is not None and c is not None and d is not None]
    if not valid:
        raise RuntimeError(f"no valid non-overlap windows for {arm}")
    a2 = np.asarray([v[0] for v in valid], dtype=float)
    a5 = np.asarray([v[1] for v in valid], dtype=float)
    a10 = np.asarray([v[2] for v in valid], dtype=float)
    af = np.asarray([v[3] for v in valid], dtype=float)
    return {
        "windows_total": int(len(rows)),
        "windows_valid_nonoverlap": int(len(valid)),
        "windows_positive_median_net2": int(np.sum(a2 > 0)),
        "positive_window_fraction_net2": float(np.mean(a2 > 0)),
        "median_weekly_phase_median_net2": float(np.median(a2)),
        "mean_weekly_phase_median_net2": float(np.mean(a2)),
        "median_weekly_phase_median_net5": float(np.median(a5)),
        "median_weekly_phase_median_net10": float(np.median(a10)),
        "min_weekly_phase_median_net2": float(np.min(a2)),
        "max_weekly_phase_median_net2": float(np.max(a2)),
        "mean_positive_phase_fraction_net2": float(np.mean(af)),
    }


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

    starts = list(pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"))
    rows = []
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
        y_test = test["target"].astype(int).to_numpy()
        if len(np.unique(y_test)) < 3:
            continue
        fwd_test = test["fwd"].to_numpy(float)
        arms = {}
        for arm in ARMS:
            train = arm_train(work, start_idx, train_end, start, arm)
            if len(train) < 2500 or len(np.unique(train["target"].astype(int).to_numpy())) < 3:
                continue
            if train["timestamp"].max() >= test["timestamp"].min():
                raise RuntimeError(f"chronology overlap {args.config_key}/{arm}/{start.date()}")
            fitted = model().fit(train[features].to_numpy(float), train["target"].astype(int).to_numpy())
            pred = fitted.predict(test[features].to_numpy(float)).astype(int)
            phase = phase_audit(test["timestamp"], pred, fwd_test, horizon)
            arms[arm] = {
                "train_rows": int(len(train)),
                "train_first_timestamp": train["timestamp"].min().isoformat(),
                "train_last_timestamp": train["timestamp"].max().isoformat(),
                "classification": classification(y_test, pred),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "nonoverlap_phase_audit": phase,
            }
            print(args.config_key, start.date(), arm, "TRAIN", len(train), "PHASE_MEDIAN_NET2", phase["median_phase_net_after_2bp"], "POS_PHASES", phase["positive_phases_net2"], "/", phase["valid_phases"])
        if set(arms) != set(ARMS):
            continue
        rows.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "arms": arms,
        })

    if len(rows) < 80:
        raise RuntimeError(f"insufficient multi-year weekly windows {len(rows)}")
    summaries = {arm: summarize(rows, arm) for arm in ARMS}
    result = {
        "schema": "foundry.nq_cc0_multiyear_nonoverlap_weekly.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "tgtanalytics/nq-futures-1min-bar-2022-2025 (CC0 Public Domain)",
        "normalized_source_sha256": source_sha,
        "config_key": args.config_key,
        "feature_set": "baseline20",
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "costs": [float(x) for x in COSTS],
        "arms": list(ARMS),
        "protocol": "predeclared seven-day future windows from 2023-07 through 2025-12; compare expanding NQ and trailing-90-calendar-day NQ logistic specialists; horizon-row purge before every test window; identical future timestamps; primary economics are all absolute-UTC non-overlapping phase streams with no phase selection; 2/5/10bp cost sensitivity reported",
        "source_rows": int(len(raw)),
        "bars_12min": int(len(bars)),
        "feature_rows": int(len(work)),
        "weekly_windows": rows,
        "summary": summaries,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_CC0_MULTIYEAR_VALIDATION=PASS")
    for arm, summary in summaries.items():
        print(arm, summary)
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
