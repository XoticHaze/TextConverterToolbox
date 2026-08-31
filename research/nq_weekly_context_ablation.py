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
TEST_END = pd.Timestamp("2025-12-12", tz="UTC")
ARMS = ("baseline20", "baseline20_plus_prior_week")

WEEKLY_FEATURES = [
    "wk_ret_1",
    "wk_range_frac",
    "wk_rv",
    "wk_volume_z8",
    "wk_close_ema4_dist",
    "wk_close_ema13_dist",
    "wk_mom4",
    "wk_mom13",
    "wk_drawdown13",
]


def monday_utc(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    return t.dt.normalize() - pd.to_timedelta(t.dt.weekday, unit="D")


def attach_prior_week_context(bars: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    x["week_start"] = monday_utc(x["timestamp"])
    pieces = []
    for week_start, g in x.groupby("week_start", sort=True):
        g = g.sort_values("timestamp")
        close = g["close"].astype(float)
        logret = np.log(close).diff()
        pieces.append({
            "week_start": week_start,
            "week_open": float(g["open"].iloc[0]),
            "week_high": float(g["high"].max()),
            "week_low": float(g["low"].min()),
            "week_close": float(g["close"].iloc[-1]),
            "week_volume": float(g["volume"].sum()),
            "week_rv": float(logret.std(ddof=0)) if logret.notna().sum() >= 10 else np.nan,
        })
    w = pd.DataFrame(pieces).sort_values("week_start").reset_index(drop=True)
    w["wk_ret_1"] = w["week_close"] / w["week_open"].replace(0, np.nan) - 1.0
    w["wk_range_frac"] = (w["week_high"] - w["week_low"]) / w["week_close"].replace(0, np.nan)
    w["wk_rv"] = w["week_rv"]
    lv = np.log1p(w["week_volume"].clip(lower=0))
    vm = lv.rolling(8, min_periods=8).mean()
    vs = lv.rolling(8, min_periods=8).std(ddof=0).replace(0, np.nan)
    w["wk_volume_z8"] = (lv - vm) / vs
    ema4 = w["week_close"].ewm(span=4, adjust=False, min_periods=4).mean()
    ema13 = w["week_close"].ewm(span=13, adjust=False, min_periods=13).mean()
    w["wk_close_ema4_dist"] = w["week_close"] / ema4 - 1.0
    w["wk_close_ema13_dist"] = w["week_close"] / ema13 - 1.0
    w["wk_mom4"] = w["week_close"].pct_change(4)
    w["wk_mom13"] = w["week_close"].pct_change(13)
    w["wk_drawdown13"] = w["week_close"] / w["week_close"].rolling(13, min_periods=13).max() - 1.0

    ctx = w[["week_start", *WEEKLY_FEATURES]].copy()
    ctx["context_source_week"] = ctx["week_start"]
    ctx["week_start"] = ctx["week_start"] + pd.Timedelta(days=7)
    out = x.merge(ctx, on="week_start", how="left", validate="many_to_one")
    bad = out["context_source_week"].notna() & ~(out["context_source_week"] < out["week_start"])
    if bad.any():
        raise RuntimeError("weekly context is not strictly prior-week")
    return out


def summarize(rows: list[dict], arm: str) -> dict:
    vals = [r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_2bp"] for r in rows]
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) < 80:
        raise RuntimeError(f"insufficient valid weekly windows for {arm}: {len(vals)}")
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
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]

    raw = load_normalized(args.nq_normalized)
    b = bars12(raw)
    b = attach_prior_week_context(b)
    frame = _add_features(b)
    feature_sets = {
        "baseline20": list(BASE_FEATURES),
        "baseline20_plus_prior_week": list(BASE_FEATURES) + WEEKLY_FEATURES,
    }
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", "week_start", "context_source_week", *feature_sets["baseline20_plus_prior_week"]]))
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
        test = work.iloc[int(idx[0]):int(idx[-1]) + 1]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["target"].notna()]
        train_end = int(idx[0]) - horizon
        if train_end < 2500 or len(test) < 300:
            continue
        train = work.iloc[:train_end]
        train = train[train["target"].notna()]
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
        raise RuntimeError(f"insufficient windows {len(rows)}")
    summary = {arm: summarize(rows, arm) for arm in ARMS}
    result = {
        "schema": "foundry.nq_prior_week_context_ablation.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "tgtanalytics/nq-futures-1min-bar-2022-2025 CC0",
        "source_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "feature_sets": {"baseline20": list(BASE_FEATURES), "baseline20_plus_prior_week": list(BASE_FEATURES) + WEEKLY_FEATURES},
        "weekly_context_contract": "every intraday row receives only the immediately preceding completed Monday-UTC week summary; current-week data is never included in weekly features",
        "weekly_windows": rows,
        "summary": summary,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_PRIOR_WEEK_CONTEXT_ABLATION=PASS")
    print(json.dumps(summary, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
