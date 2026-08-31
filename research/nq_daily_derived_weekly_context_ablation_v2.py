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
from research.nq_daily_derived_weekly_context_ablation import CONFIGS, WEEKLY_FEATURES, load_daily, summarize

TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-10-06", tz="UTC")
ARMS = ("baseline20", "daily_derived_weekly_26y")


def futures_trade_week_key(ts: pd.Series) -> pd.Series:
    """Return naive local-Monday trade-week keys, stable across DST transitions."""
    t = pd.to_datetime(ts, utc=True, errors="raise")
    local = t.dt.tz_convert("America/New_York")
    wall = local.dt.tz_localize(None)
    trade_date = wall.dt.normalize() + pd.to_timedelta((wall.dt.hour >= 18).astype(int), unit="D")
    return trade_date - pd.to_timedelta(trade_date.dt.weekday, unit="D")


def derive_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["source_week_key"] = futures_trade_week_key(d["timestamp"])
    weekly = d.groupby("source_week_key", sort=True).agg(
        week_open=("open", "first"), week_high=("high", "max"), week_low=("low", "min"),
        week_close=("close", "last"), week_volume=("volume", "sum"), daily_bars=("close", "count"),
        first_daily_ts=("timestamp", "min"), last_daily_ts=("timestamp", "max"),
    ).reset_index()
    weekly = weekly[weekly["daily_bars"] >= 3].reset_index(drop=True)
    if weekly["source_week_key"].duplicated().any():
        raise RuntimeError("derived weekly source keys not unique")

    close = weekly["week_close"].astype(float)
    logret = np.log(close).diff()
    for lag in (1, 4, 13, 26, 52):
        weekly[f"dw_ret{lag}"] = close.pct_change(lag)
    weekly["dw_range_frac"] = (weekly["week_high"] - weekly["week_low"]) / close.replace(0, np.nan)
    for window in (13, 26, 52):
        weekly[f"dw_rv{window}"] = logret.rolling(window, min_periods=window).std(ddof=0)
    lv = np.log1p(weekly["week_volume"].clip(lower=0))
    weekly["dw_volume_z52"] = (lv - lv.rolling(52, min_periods=52).mean()) / lv.rolling(52, min_periods=52).std(ddof=0).replace(0, np.nan)
    for span in (13, 26, 52):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        weekly[f"dw_ema{span}_dist"] = close / ema - 1.0
    weekly["dw_drawdown52"] = close / close.rolling(52, min_periods=52).max() - 1.0

    weekly["trade_week_key"] = weekly["source_week_key"] + pd.Timedelta(days=7)
    if not (weekly["source_week_key"] < weekly["trade_week_key"]).all():
        raise RuntimeError("derived weekly context availability is not lagged")
    return weekly[["trade_week_key", "source_week_key", "daily_bars", *WEEKLY_FEATURES]]


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
    b["trade_week_key"] = futures_trade_week_key(b["timestamp"])
    weekly = derive_weekly(load_daily(args.nq_daily))
    b = b.merge(weekly, on="trade_week_key", how="left", validate="many_to_one")
    if (b["source_week_key"].notna() & ~(b["source_week_key"] < b["trade_week_key"])).any():
        raise RuntimeError("derived weekly context leak")

    frame = _add_features(b)
    feature_sets = {
        "baseline20": list(BASE_FEATURES),
        "daily_derived_weekly_26y": list(BASE_FEATURES) + WEEKLY_FEATURES,
    }
    all_features = feature_sets["daily_derived_weekly_26y"]
    eligible = frame[frame["timestamp"] < TEST_END].copy()
    base = eligible[["timestamp", "close", "rv_120", *BASE_FEATURES]].replace([np.inf, -np.inf], np.nan).dropna()
    enriched = eligible[["timestamp", "close", "rv_120", "trade_week_key", *all_features]].replace([np.inf, -np.inf], np.nan).dropna()
    missing = pd.Index(base["timestamp"]).difference(pd.Index(enriched["timestamp"]))
    if len(missing):
        raise RuntimeError(f"daily-derived context discards baseline rows: {len(missing)}")
    if not pd.Index(base["timestamp"]).equals(pd.Index(enriched["timestamp"])):
        raise RuntimeError("baseline/enriched row ordering differs")

    work = enriched.reset_index(drop=True)
    label, fwd, _ = target_columns(work, horizon, vol_mult)
    work["target"] = label
    work["fwd"] = fwd

    rows = []
    for start in pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"):
        end = start + pd.Timedelta(days=7)
        idx = np.flatnonzero(((work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna()).to_numpy())
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
        if len(test) < 300 or train["timestamp"].max() >= test["timestamp"].min():
            continue
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
        "schema": "foundry.nq_daily_derived_weekly_context_ablation.v2",
        "research_only": True,
        "promotion_authority": False,
        "intraday_source": "tgtanalytics/nq-futures-1min-bar-2022-2025 CC0",
        "daily_source": "youneseloiarm/nasdaq-cme-future-nq:NQ_in_daily.csv CC0 1999+",
        "intraday_source_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "daily_source_sha256": hashlib.sha256(args.nq_daily.read_bytes()).hexdigest(),
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "session_contract": "America/New_York futures trade date; wall times at/after 18:00 belong to next trade date; join key is naive local Monday date so DST cannot shift week identity; completed source week publishes only to following trade week",
        "protocol": "baseline20 versus 1999+ weekly state derived from unique daily bars on identical intraday rows; horizon purge before each weekly OOS window; all non-overlapping UTC phase streams reported; direct vendor weekly labels excluded",
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
