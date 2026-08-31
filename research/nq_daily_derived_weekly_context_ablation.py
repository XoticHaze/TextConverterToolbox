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

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-10-06", tz="UTC")
ARMS = ("baseline20", "daily_derived_weekly_26y")

WEEKLY_FEATURES = [
    "dw_ret1", "dw_ret4", "dw_ret13", "dw_ret26", "dw_ret52",
    "dw_range_frac", "dw_rv13", "dw_rv26", "dw_rv52",
    "dw_volume_z52", "dw_ema13_dist", "dw_ema26_dist", "dw_ema52_dist",
    "dw_drawdown52",
]


def futures_trade_week_start(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    local = t.dt.tz_convert("America/New_York")
    local_date = local.dt.normalize()
    trade_date = local_date + pd.to_timedelta((local.dt.hour >= 18).astype(int), unit="D")
    weekday = trade_date.dt.weekday
    week = trade_date - pd.to_timedelta(weekday, unit="D")
    return week.dt.tz_convert("UTC")


def load_daily(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in f.columns]
    if missing:
        raise RuntimeError(f"missing daily columns {missing}")
    f["timestamp"] = pd.to_datetime(f["datetime"], utc=True, errors="raise")
    for c in required[1:]:
        f[c] = pd.to_numeric(f[c], errors="raise")
    f = f.sort_values("timestamp").reset_index(drop=True)
    if f["timestamp"].duplicated().any() or not f["timestamp"].is_monotonic_increasing:
        raise RuntimeError("daily timestamps not unique/increasing")
    return f[["timestamp", "open", "high", "low", "close", "volume"]]


def derive_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["source_week_start"] = futures_trade_week_start(d["timestamp"])
    weekly = d.groupby("source_week_start", sort=True).agg(
        week_open=("open", "first"),
        week_high=("high", "max"),
        week_low=("low", "min"),
        week_close=("close", "last"),
        week_volume=("volume", "sum"),
        daily_bars=("close", "count"),
        first_daily_ts=("timestamp", "min"),
        last_daily_ts=("timestamp", "max"),
    ).reset_index()
    weekly = weekly[weekly["daily_bars"] >= 3].reset_index(drop=True)
    if weekly["source_week_start"].duplicated().any():
        raise RuntimeError("derived weekly source keys not unique")

    close = weekly["week_close"].astype(float)
    logret = np.log(close).diff()
    for lag in (1, 4, 13, 26, 52):
        weekly[f"dw_ret{lag}"] = close.pct_change(lag)
    weekly["dw_range_frac"] = (weekly["week_high"] - weekly["week_low"]) / close.replace(0, np.nan)
    for window in (13, 26, 52):
        weekly[f"dw_rv{window}"] = logret.rolling(window, min_periods=window).std(ddof=0)
    lv = np.log1p(weekly["week_volume"].clip(lower=0))
    vm = lv.rolling(52, min_periods=52).mean()
    vs = lv.rolling(52, min_periods=52).std(ddof=0).replace(0, np.nan)
    weekly["dw_volume_z52"] = (lv - vm) / vs
    for span in (13, 26, 52):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        weekly[f"dw_ema{span}_dist"] = close / ema - 1.0
    weekly["dw_drawdown52"] = close / close.rolling(52, min_periods=52).max() - 1.0

    # The source week's OHLCV is only known after that trade week finishes.
    # Publish its state to the following trade week, never the same week.
    weekly["trade_week_start"] = weekly["source_week_start"] + pd.Timedelta(days=7)
    if not (weekly["source_week_start"] < weekly["trade_week_start"]).all():
        raise RuntimeError("derived weekly context availability is not lagged")
    return weekly[[
        "trade_week_start", "source_week_start", "daily_bars", "first_daily_ts", "last_daily_ts",
        *WEEKLY_FEATURES,
    ]]


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
    ap.add_argument("--nq-daily", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]

    intraday = load_normalized(args.nq_normalized)
    b = bars12(intraday)
    b["trade_week_start"] = futures_trade_week_start(b["timestamp"])
    weekly = derive_weekly(load_daily(args.nq_daily))
    b = b.merge(weekly, on="trade_week_start", how="left", validate="many_to_one")
    bad = b["source_week_start"].notna() & ~(b["source_week_start"] < b["trade_week_start"])
    if bad.any():
        raise RuntimeError("derived weekly context leak")

    frame = _add_features(b)
    feature_sets = {
        "baseline20": list(BASE_FEATURES),
        "daily_derived_weekly_26y": list(BASE_FEATURES) + WEEKLY_FEATURES,
    }
    all_features = feature_sets["daily_derived_weekly_26y"]
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", "trade_week_start", *all_features]))
    eligible = frame[frame["trade_week_start"] < TEST_END].copy()

    baseline_panel = eligible[["timestamp", "close", "rv_120", *BASE_FEATURES]].replace([np.inf, -np.inf], np.nan).dropna()
    enriched_panel = eligible[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if not pd.Index(baseline_panel["timestamp"]).equals(pd.Index(enriched_panel["timestamp"])):
        missing = pd.Index(baseline_panel["timestamp"]).difference(pd.Index(enriched_panel["timestamp"]))
        raise RuntimeError(f"long daily-derived context discards baseline rows: {len(missing)}")

    work = enriched_panel.reset_index(drop=True)
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
        test_start, test_stop = int(idx[0]), int(idx[-1] + 1)
        train_end = test_start - horizon
        if train_end < 2500:
            continue
        train = work.iloc[:train_end]
        train = train[train["target"].notna()]
        test = work.iloc[test_start:test_stop]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["target"].notna()]
        if len(test) < 300:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError(f"chronology overlap {start.date()}")
        y_test = test["target"].astype(int).to_numpy()
        fwd_test = test["fwd"].to_numpy(float)
        arms = {}
        for arm, features in feature_sets.items():
            fitted = model().fit(train[features].to_numpy(float), train["target"].astype(int).to_numpy())
            pred = fitted.predict(test[features].to_numpy(float)).astype(int)
            arms[arm] = {
                "train_rows": int(len(train)),
                "classification": classification(y_test, pred),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "nonoverlap_phase_audit": phase_audit(test["timestamp"], pred, fwd_test, horizon),
            }
        rows.append({"start": start.isoformat(), "end": end.isoformat(), "test_rows": int(len(test)), "arms": arms})

    if len(rows) < 80:
        raise RuntimeError(f"insufficient comparable windows {len(rows)}")
    summary = {arm: summarize(rows, arm) for arm in ARMS}
    result = {
        "schema": "foundry.nq_daily_derived_weekly_context_ablation.v1",
        "research_only": True,
        "promotion_authority": False,
        "intraday_source": "tgtanalytics/nq-futures-1min-bar-2022-2025 CC0",
        "daily_source": "youneseloiarm/nasdaq-cme-future-nq:NQ_in_daily.csv CC0 1999+",
        "intraday_source_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "daily_source_sha256": hashlib.sha256(args.nq_daily.read_bytes()).hexdigest(),
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "session_contract": "CME-style futures trade date derived in America/New_York: timestamps at/after 18:00 ET belong to the following trade date; trade weeks begin Monday; completed weekly OHLCV is published only to the following trade week",
        "protocol": "baseline20 versus 1999+ slow weekly NQ state derived deterministically from unique daily bars; identical intraday rows and weekly OOS timestamps; horizon purge before every future week; all non-overlapping UTC phase streams reported; no direct vendor-weekly labels used",
        "feature_sets": feature_sets,
        "weekly_windows": rows,
        "summary": summary,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_DAILY_DERIVED_WEEKLY_CONTEXT_ABLATION=PASS")
    print(json.dumps(summary, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
