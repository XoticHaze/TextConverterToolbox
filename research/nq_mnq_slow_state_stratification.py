from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_nq_domain_adaptation import fit_equal_market
from research.mnq_opportunity_target_matrix import classification, model, target_columns
from research.nq_cc0_multiyear_validation import load_normalized, bars12
from research.nq_to_mnq_execution_transfer import phase_audit

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-12-08", tz="UTC")
ARMS = ("nq_expanding", "mnq_expanding", "equal_market_pooled", "always_long", "always_short")
POINT_FIELD = "median_phase_net_points_after_1p0pt"


def trade_week_key(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    local = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    trade_date = local.dt.normalize() + pd.to_timedelta((local.dt.hour >= 18).astype(int), unit="D")
    return trade_date - pd.to_timedelta(trade_date.dt.weekday, unit="D")


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


def slow_state_table(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["source_week"] = trade_week_key(d["timestamp"])
    w = d.groupby("source_week", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), daily_bars=("close", "count"),
    ).reset_index()
    w = w[w["daily_bars"] >= 3].reset_index(drop=True)
    if w["source_week"].duplicated().any():
        raise RuntimeError("derived weekly keys not unique")

    close = w["close"].astype(float)
    lr = np.log(close).diff()
    w["ret13"] = close.pct_change(13)
    w["ret26"] = close.pct_change(26)
    w["ret52"] = close.pct_change(52)
    w["rv13"] = lr.rolling(13, min_periods=13).std(ddof=0)
    w["rv52"] = lr.rolling(52, min_periods=52).std(ddof=0)
    ema13 = close.ewm(span=13, adjust=False, min_periods=13).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ema52 = close.ewm(span=52, adjust=False, min_periods=52).mean()
    w["ema13_dist"] = close / ema13 - 1.0
    w["ema26_dist"] = close / ema26 - 1.0
    w["ema52_dist"] = close / ema52 - 1.0
    w["drawdown52"] = close / close.rolling(52, min_periods=52).max() - 1.0
    w["vol_ratio_13_52"] = w["rv13"] / w["rv52"].replace(0, np.nan)

    def trend(row) -> str:
        if row["ret13"] > 0 and row["ema13_dist"] > 0 and row["ema26_dist"] > 0:
            return "bull"
        if row["ret13"] < 0 and row["ema13_dist"] < 0 and row["ema26_dist"] < 0:
            return "bear"
        return "mixed"

    def vol(row) -> str:
        x = row["vol_ratio_13_52"]
        if not np.isfinite(x):
            return "unknown"
        if x >= 1.25:
            return "high"
        if x <= 0.80:
            return "low"
        return "normal"

    def dd(row) -> str:
        x = row["drawdown52"]
        if not np.isfinite(x):
            return "unknown"
        if x <= -0.10:
            return "deep"
        if x <= -0.05:
            return "moderate"
        return "shallow"

    w["trend_state"] = w.apply(trend, axis=1)
    w["vol_state"] = w.apply(vol, axis=1)
    w["drawdown_state"] = w.apply(dd, axis=1)
    # State from a completed source week is first available to the following trade week.
    w["trade_week"] = w["source_week"] + pd.Timedelta(days=7)
    return w[["trade_week", "source_week", "trend_state", "vol_state", "drawdown_state",
              "ret13", "ret26", "ret52", "vol_ratio_13_52", "drawdown52", "ema52_dist"]]


def purged_train(frame: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    before = frame[(frame["timestamp"] < start) & frame["target"].notna()].copy()
    if len(before) <= horizon:
        return before.iloc[0:0]
    return before.iloc[:-horizon].copy()


def summarize_state(rows: list[dict], dim: str) -> dict:
    out: dict[str, dict] = {}
    states = sorted({r[dim] for r in rows})
    for state in states:
        subset = [r for r in rows if r[dim] == state]
        if len(subset) < 6:
            continue
        arms = {}
        for arm in ARMS:
            vals = np.asarray([r["arms"][arm]["phase_audit"].get(POINT_FIELD) for r in subset], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 6:
                continue
            arms[arm] = {
                "weeks": int(len(vals)),
                "median_net_points_after_1pt": float(np.median(vals)),
                "mean_net_points_after_1pt": float(np.mean(vals)),
                "positive_week_fraction": float(np.mean(vals > 0)),
            }
        paired = {}
        if "mnq_expanding" in arms:
            for challenger in ("nq_expanding", "equal_market_pooled"):
                diffs = []
                for r in subset:
                    a = r["arms"][challenger]["phase_audit"].get(POINT_FIELD)
                    b = r["arms"]["mnq_expanding"]["phase_audit"].get(POINT_FIELD)
                    if a is not None and b is not None:
                        diffs.append(float(a) - float(b))
                arr = np.asarray(diffs, dtype=float)
                if len(arr) >= 6:
                    paired[challenger + "_minus_mnq"] = {
                        "weeks": int(len(arr)),
                        "median_delta_points": float(np.median(arr)),
                        "mean_delta_points": float(np.mean(arr)),
                        "win_fraction": float(np.mean(arr > 0)),
                    }
        out[state] = {"weeks": len(subset), "arms": arms, "paired": paired}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--nq-normalized", type=Path, required=True)
    ap.add_argument("--nq-daily", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]
    features = list(BASE_FEATURES)

    mnq_raw = load_deep(args.deep_root)
    mnq = _add_features(deep_bars(stitch_deep(mnq_raw, deep_roll_schedule(mnq_raw))))
    nq = _add_features(bars12(load_normalized(args.nq_normalized)))
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    mnq_target, _, _ = target_columns(mnq, horizon, vol_mult)
    nq_target, _, _ = target_columns(nq, horizon, vol_mult)
    mnq["target"] = mnq_target
    nq["target"] = nq_target
    mnq["point_move"] = mnq["close"].shift(-horizon) - mnq["close"]

    state = slow_state_table(load_daily(args.nq_daily))
    state_map = {pd.Timestamp(r.trade_week): r for r in state.itertuples(index=False)}
    starts = list(pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"))
    rows = []
    for start in starts:
        end = start + pd.Timedelta(days=7)
        local_key = trade_week_key(pd.Series([start]))[0]
        state_row = state_map.get(pd.Timestamp(local_key))
        if state_row is None or state_row.trend_state == "unknown" or state_row.vol_state == "unknown":
            continue
        test = mnq[(mnq["timestamp"] >= start) & (mnq["timestamp"] < end) & mnq["target"].notna() & mnq["point_move"].notna()].copy()
        if len(test) < 300:
            continue
        mnq_train = purged_train(mnq, start, horizon)
        nq_train = purged_train(nq, start, horizon)
        if len(mnq_train) < 50000 or len(nq_train) < 10000:
            continue
        if mnq_train["timestamp"].max() >= test["timestamp"].min() or nq_train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("chronology overlap")
        fitted = {
            "nq_expanding": model().fit(nq_train[features].to_numpy(float), nq_train["target"].astype(int).to_numpy()),
            "mnq_expanding": model().fit(mnq_train[features].to_numpy(float), mnq_train["target"].astype(int).to_numpy()),
            "equal_market_pooled": fit_equal_market(features, mnq_train, nq_train),
        }
        y = test["target"].astype(int).to_numpy()
        move = test["point_move"].to_numpy(float)
        arms = {}
        for arm, fitted_model in fitted.items():
            pred = fitted_model.predict(test[features].to_numpy(float)).astype(int)
            arms[arm] = {
                "classification": classification(y, pred),
                "phase_audit": phase_audit(test["timestamp"], pred, move, horizon),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
            }
        for arm, pred in {"always_long": np.ones(len(test), dtype=int), "always_short": -np.ones(len(test), dtype=int)}.items():
            arms[arm] = {"classification": classification(y, pred), "phase_audit": phase_audit(test["timestamp"], pred, move, horizon)}
        rows.append({
            "start": start.isoformat(), "end": end.isoformat(), "rows": int(len(test)),
            "trend_state": state_row.trend_state, "vol_state": state_row.vol_state,
            "drawdown_state": state_row.drawdown_state,
            "state_source_week": pd.Timestamp(state_row.source_week).isoformat(),
            "state_values": {"ret13": float(state_row.ret13), "ret26": float(state_row.ret26),
                             "ret52": float(state_row.ret52), "vol_ratio_13_52": float(state_row.vol_ratio_13_52),
                             "drawdown52": float(state_row.drawdown52), "ema52_dist": float(state_row.ema52_dist)},
            "arms": arms,
        })

    if len(rows) < 80:
        raise RuntimeError(f"insufficient state-audited weeks {len(rows)}")
    result = {
        "schema": "foundry.nq_mnq_slow_state_stratification.v1",
        "research_only": True,
        "promotion_authority": False,
        "execution_target": "MNQ",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "feature_set": "baseline20 only; slow state is diagnostic and is never fed into prediction",
        "state_contract": "NQ daily bars aggregated to futures trade weeks under 18:00 America/New_York trade-date boundary; state from each completed week becomes available only to the next trade week; fixed trend/vol/drawdown definitions are predeclared and do not use OOS PnL",
        "protocol": "reproduce weekly past-only NQ-only, MNQ-only, pooled and directional baselines on MNQ OOS; attach prior completed NQ slow state after predictions; report economics by state without gating, threshold fitting, phase selection, or prediction changes",
        "weekly_rows": rows,
        "stratification": {
            "trend_state": summarize_state(rows, "trend_state"),
            "vol_state": summarize_state(rows, "vol_state"),
            "drawdown_state": summarize_state(rows, "drawdown_state"),
        },
        "nq_normalized_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "nq_daily_sha256": hashlib.sha256(args.nq_daily.read_bytes()).hexdigest(),
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_MNQ_SLOW_STATE_STRATIFICATION=PASS")
    print(json.dumps(result["stratification"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
