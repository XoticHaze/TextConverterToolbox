from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_opportunity_target_matrix import classification, model, target_columns
from research.nq_cc0_multiyear_validation import load_normalized, bars12, phase_audit
from research.nq_weekly_context_ablation import WEEKLY_FEATURES, attach_prior_week_context

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-10-06", tz="UTC")
ARMS = ("baseline20", "short_prior_week", "long_weekly_26y")

LONG_WEEKLY_FEATURES = [
    "lw_ret1", "lw_ret4", "lw_ret13", "lw_ret26", "lw_ret52",
    "lw_range_frac", "lw_rv13", "lw_rv26", "lw_rv52",
    "lw_volume_z52", "lw_ema13_dist", "lw_ema26_dist", "lw_ema52_dist",
    "lw_drawdown52",
]


def week_start_utc(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    return t.dt.normalize() - pd.to_timedelta(t.dt.weekday, unit="D")


def load_long_weekly(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in f.columns]
    if missing:
        raise RuntimeError(f"missing long weekly columns {missing}")
    f["timestamp"] = pd.to_datetime(f["datetime"], utc=True, errors="raise")
    for c in required[1:]:
        f[c] = pd.to_numeric(f[c], errors="raise")
    f = f.sort_values("timestamp").drop_duplicates("timestamp", keep=False).reset_index(drop=True)
    if f["timestamp"].duplicated().any() or not f["timestamp"].is_monotonic_increasing:
        raise RuntimeError("long weekly timestamps not unique/increasing")
    return f[["timestamp", "open", "high", "low", "close", "volume"]]


def long_context_table(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy().sort_values("timestamp").reset_index(drop=True)
    close = w["close"].astype(float)
    logret = np.log(close).diff()
    for lag in (1, 4, 13, 26, 52):
        w[f"lw_ret{lag}"] = close.pct_change(lag)
    w["lw_range_frac"] = (w["high"] - w["low"]) / close.replace(0, np.nan)
    for window in (13, 26, 52):
        w[f"lw_rv{window}"] = logret.rolling(window, min_periods=window).std(ddof=0)
    lv = np.log1p(w["volume"].clip(lower=0))
    vm = lv.rolling(52, min_periods=52).mean()
    vs = lv.rolling(52, min_periods=52).std(ddof=0).replace(0, np.nan)
    w["lw_volume_z52"] = (lv - vm) / vs
    for span in (13, 26, 52):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        w[f"lw_ema{span}_dist"] = close / ema - 1.0
    w["lw_drawdown52"] = close / close.rolling(52, min_periods=52).max() - 1.0

    # Source weekly bars are timestamped near the START of their week. Their OHLCV is
    # unavailable until that week has finished, so publish the state to the NEXT week only.
    source_week = week_start_utc(w["timestamp"])
    out = w[LONG_WEEKLY_FEATURES].copy()
    out["long_context_source_week"] = source_week
    out["week_start"] = source_week + pd.Timedelta(days=7)
    if not (out["long_context_source_week"] < out["week_start"]).all():
        raise RuntimeError("long weekly context availability is not lagged")
    return out[["week_start", "long_context_source_week", *LONG_WEEKLY_FEATURES]]


def summarize(rows: list[dict], arm: str) -> dict:
    vals = [r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_2bp"] for r in rows]
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) < 80:
        raise RuntimeError(f"insufficient valid windows {arm}={len(vals)}")
    return {
        "windows": int(len(vals)),
        "positive_windows_net2": int(np.sum(vals > 0)),
        "positive_window_fraction_net2": float(np.mean(vals > 0)),
        "median_weekly_phase_median_net2": float(np.median(vals)),
        "mean_weekly_phase_median_net2": float(np.mean(vals)),
        "min_weekly_phase_median_net2": float(np.min(vals)),
        "max_weekly_phase_median_net2": float(np.max(vals)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--nq-normalized", type=Path, required=True)
    ap.add_argument("--nq-weekly", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]

    intraday = load_normalized(args.nq_normalized)
    b = bars12(intraday)
    b = attach_prior_week_context(b)
    b["week_start"] = week_start_utc(b["timestamp"])
    long_ctx = long_context_table(load_long_weekly(args.nq_weekly))
    b = b.merge(long_ctx, on="week_start", how="left", validate="many_to_one")
    bad = b["long_context_source_week"].notna() & ~(b["long_context_source_week"] < b["week_start"])
    if bad.any():
        raise RuntimeError("long weekly context leak")

    frame = _add_features(b)
    feature_sets = {
        "baseline20": list(BASE_FEATURES),
        "short_prior_week": list(BASE_FEATURES) + WEEKLY_FEATURES,
        "long_weekly_26y": list(BASE_FEATURES) + LONG_WEEKLY_FEATURES,
    }
    all_features = list(dict.fromkeys(sum(feature_sets.values(), [])))
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", "week_start", *all_features]))
    work = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, fwd, _ = target_columns(work, horizon, vol_mult)
    work["target"] = label
    work["fwd"] = fwd

    starts = list(pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"))
    rows = []
    for start in starts:
        end = start + pd.Timedelta(days=7)
        mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna()
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 300:
            continue
        train_end = int(idx[0]) - horizon
        if train_end < 2500:
            continue
        train = work.iloc[:train_end]
        train = train[train["target"].notna()]
        test = work.iloc[int(idx[0]):int(idx[-1]) + 1]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["target"].notna()]
        if len(test) < 300:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("chronology overlap")
        y_test = test["target"].astype(int).to_numpy()
        fwd_test = test["fwd"].to_numpy(float)
        arms = {}
        for arm, features in feature_sets.items():
            fitted = model().fit(train[features].to_numpy(float), train["target"].astype(int).to_numpy())
            pred = fitted.predict(test[features].to_numpy(float)).astype(int)
            arms[arm] = {
                "classification": classification(y_test, pred),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "nonoverlap_phase_audit": phase_audit(test["timestamp"], pred, fwd_test, horizon),
            }
        rows.append({"start": start.isoformat(), "end": end.isoformat(), "test_rows": int(len(test)), "arms": arms})

    if len(rows) < 80:
        raise RuntimeError(f"insufficient comparable windows {len(rows)}")
    summary = {arm: summarize(rows, arm) for arm in ARMS}
    result = {
        "schema": "foundry.nq_long_weekly_context_ablation.v1",
        "research_only": True,
        "promotion_authority": False,
        "intraday_source": "tgtanalytics/nq-futures-1min-bar-2022-2025 CC0",
        "long_weekly_source": "youneseloiarm/nasdaq-cme-future-nq:NQ_in_weekly.csv CC0",
        "intraday_source_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "weekly_source_sha256": hashlib.sha256(args.nq_weekly.read_bytes()).hexdigest(),
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "protocol": "same fixed weekly OOS timestamps for baseline20, intraday-derived immediately-prior-week state, and independent 1999+ weekly-source slow state; all weekly OHLCV features are withheld until the following week; horizon purge before every test window; all non-overlapping UTC phase streams reported with no phase selection",
        "feature_sets": feature_sets,
        "weekly_windows": rows,
        "summary": summary,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_LONG_WEEKLY_CONTEXT_ABLATION=PASS")
    print(json.dumps(summary, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
