from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep

SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
TEST_START = pd.Timestamp("2022-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
CONFIGS = {
    "m1_h30_v05": {"bar_minutes": 1, "horizon": 30, "vol_multiplier": 0.5, "model": "sgd"},
    "m1_h60_v05": {"bar_minutes": 1, "horizon": 60, "vol_multiplier": 0.5, "model": "sgd"},
    "h1_h4_v05": {"bar_minutes": 60, "horizon": 4, "vol_multiplier": 0.5, "model": "logistic"},
    "h1_h12_v05": {"bar_minutes": 60, "horizon": 12, "vol_multiplier": 0.5, "model": "logistic"},
}
FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "range_frac", "body_frac", "log_volume", "volume_z_24", "volume_z_120",
    "rv_12", "rv_60", "rv_120", "ema12_dist", "ema48_dist",
    "mom_24", "mom_120", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def bars_for_resolution(stitched: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    if bar_minutes == 1:
        out = stitched[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].copy()
        out = out.rename(columns={"symbol": "source_contract"})
        out["observed_minutes"] = 1
        out["market"] = "MNQ"
        return out.reset_index(drop=True)

    w = stitched.set_index("timestamp")
    rule = f"{bar_minutes}min"
    out = w.resample(rule, origin="start_day", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        observed_minutes=("close", "count"),
        source_contract=("symbol", "first"),
        source_contract_last=("symbol", "last"),
    )
    out = out[out["observed_minutes"] > 0].copy()
    mixed = out["source_contract"] != out["source_contract_last"]
    out = out.loc[~mixed].drop(columns=["source_contract_last"]).reset_index()
    out["market"] = "MNQ"
    out.attrs["dropped_roll_boundary_bars"] = int(mixed.sum())
    return out


def future_extreme(series: pd.Series, horizon: int, kind: str) -> pd.Series:
    shifted = pd.concat([series.shift(-i) for i in range(1, horizon + 1)], axis=1)
    if kind == "max":
        return shifted.max(axis=1, skipna=False)
    if kind == "min":
        return shifted.min(axis=1, skipna=False)
    raise ValueError(kind)


def segment_features(group: pd.DataFrame, horizon: int, vol_multiplier: float) -> pd.DataFrame:
    g = group.copy().reset_index(drop=True)
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    open_ = g["open"].astype(float)
    log_close = np.log(close)
    ret1 = log_close.diff()

    for lag in (1, 3, 6, 12):
        g[f"ret_{lag}"] = log_close.diff(lag)
    g["range_frac"] = (high - low) / close.replace(0, np.nan)
    g["body_frac"] = (close - open_) / open_.replace(0, np.nan)
    g["log_volume"] = np.log1p(g["volume"].astype(float).clip(lower=0))
    for window in (24, 120):
        mean = g["log_volume"].rolling(window, min_periods=window).mean()
        std = g["log_volume"].rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
        g[f"volume_z_{window}"] = (g["log_volume"] - mean) / std
    for window in (12, 60, 120):
        g[f"rv_{window}"] = ret1.rolling(window, min_periods=window).std(ddof=0)
    for span in (12, 48):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        g[f"ema{span}_dist"] = close / ema - 1.0
    g["mom_24"] = close.pct_change(24)
    g["mom_120"] = close.pct_change(120)

    hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
    g["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    g["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    dow = g["timestamp"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * math.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * math.pi * dow / 7.0)

    future_close = close.shift(-horizon)
    future_high = future_extreme(high, horizon, "max")
    future_low = future_extreme(low, horizon, "min")
    fwd_log = np.log(future_close / close)
    threshold = vol_multiplier * g["rv_120"] * math.sqrt(horizon)
    valid = future_close.notna() & threshold.notna()
    target = pd.Series(np.nan, index=g.index, dtype=float)
    target.loc[valid] = 0.0
    target.loc[valid & (fwd_log > threshold)] = 1.0
    target.loc[valid & (fwd_log < -threshold)] = -1.0
    g["target"] = target
    g["point_move"] = future_close - close
    scale = close * g["rv_120"] * math.sqrt(horizon)
    g["long_mae_z"] = (close - future_low).clip(lower=0) / scale.replace(0, np.nan)
    g["short_mae_z"] = (future_high - close).clip(lower=0) / scale.replace(0, np.nan)
    return g


def build_matrix(bars: pd.DataFrame, horizon: int, vol_multiplier: float) -> pd.DataFrame:
    work = bars.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    segment = work["source_contract"].ne(work["source_contract"].shift()).cumsum()
    parts = [segment_features(g, horizon, vol_multiplier) for _, g in work.groupby(segment, sort=False)]
    out = pd.concat(parts, ignore_index=True)
    keep = [
        "timestamp", "source_contract", "close", *FEATURES,
        "target", "point_move", "long_mae_z", "short_mae_z",
    ]
    out = out[keep].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[*FEATURES, "target", "point_move"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    if out["timestamp"].duplicated().any() or not out["timestamp"].is_monotonic_increasing:
        raise RuntimeError("matrix timestamps are not unique/increasing")
    return out


def model_for(name: str) -> Pipeline:
    if name == "sgd":
        return Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=1500,
                    tol=1e-3,
                    class_weight="balanced",
                    average=True,
                    random_state=42,
                ),
            ),
        ])
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
        ])
    raise ValueError(name)


def trade_week_key(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    local = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    trade_date = local.dt.normalize() + pd.to_timedelta((local.dt.hour >= 18).astype(int), unit="D")
    return trade_date - pd.to_timedelta(trade_date.dt.weekday, unit="D")


def phase_audit(ts: pd.Series, pred: np.ndarray, point_move: np.ndarray, bar_minutes: int, horizon: int) -> dict:
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    slots = ((parsed - epoch) // pd.Timedelta(minutes=bar_minutes)).to_numpy(dtype=np.int64)
    min_signals = 4 if bar_minutes >= 60 else 10
    phases = {}
    for phase in range(horizon):
        mask = (slots % horizon) == phase
        p = pred[mask]
        m = point_move[mask]
        selected = p != 0
        side = np.where(p[selected] == 1, 1.0, -1.0)
        gross = side * m[selected]
        rec = {"signals": int(len(gross))}
        if len(gross) >= min_signals:
            rec["gross_mean_points"] = float(np.mean(gross))
            for cost in POINT_COSTS:
                key = str(cost).replace(".", "p")
                rec[f"net_mean_points_after_{key}pt"] = float(np.mean(gross - cost))
        phases[str(phase)] = rec
    valid = [v for v in phases.values() if "net_mean_points_after_1p0pt" in v]
    out = {
        "valid_phases": int(len(valid)),
        "phase_streams": phases,
        "contract": f"absolute UTC {bar_minutes}min slot modulo H{horizon}; every phase reported; no post-hoc phase selection",
    }
    if valid:
        for cost in POINT_COSTS:
            key = str(cost).replace(".", "p")
            vals = np.asarray([float(v[f"net_mean_points_after_{key}pt"]) for v in valid], dtype=float)
            out[f"median_phase_net_points_after_{key}pt"] = float(np.median(vals))
            out[f"mean_phase_net_points_after_{key}pt"] = float(np.mean(vals))
            out[f"positive_phase_fraction_after_{key}pt"] = float(np.mean(vals > 0))
    return out


def classification(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "coverage": float(np.mean(pred != 0)),
        "predicted_counts": {str(c): int(np.sum(pred == c)) for c in (-1, 0, 1)},
    }


def economic_summary(rows: list[dict], policy: str, cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    vals = []
    for row in rows:
        value = row["policies"][policy]["phase_audit"].get(field)
        if value is not None and np.isfinite(value):
            vals.append(float(value))
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return {"weeks": 0}
    k = max(1, int(np.ceil(0.10 * len(arr))))
    worst = np.sort(arr)[:k]
    cumulative = np.cumsum(arr)
    curve = np.concatenate(([0.0], cumulative))
    peak = np.maximum.accumulate(curve)
    drawdown = curve - peak
    return {
        "weeks": int(len(arr)),
        "positive_weeks": int(np.sum(arr > 0)),
        "positive_week_fraction": float(np.mean(arr > 0)),
        "median_weekly_phase_median_points": float(np.median(arr)),
        "mean_weekly_phase_median_points": float(np.mean(arr)),
        "p10_weekly_phase_median_points": float(np.quantile(arr, 0.10)),
        "bottom10pct_mean_points": float(np.mean(worst)),
        "worst_week_points": float(np.min(arr)),
        "best_week_points": float(np.max(arr)),
        "cumulative_weekly_phase_median_points": float(np.sum(arr)),
        "max_drawdown_weekly_phase_median_points": float(np.min(drawdown)),
    }


def decomposition(rows: list[dict], policy: str, cost: float, field: str) -> dict:
    key = str(cost).replace(".", "p")
    metric = f"median_phase_net_points_after_{key}pt"
    groups = {}
    for row in rows:
        label = row[field]
        value = row["policies"][policy]["phase_audit"].get(metric)
        if value is not None and np.isfinite(value):
            groups.setdefault(label, []).append(float(value))
    return {
        label: {
            "weeks": len(vals),
            "median_points": float(np.median(vals)),
            "mean_points": float(np.mean(vals)),
            "positive_fraction": float(np.mean(np.asarray(vals) > 0)),
        }
        for label, vals in sorted(groups.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = CONFIGS[args.config_key]
    bar_minutes = int(cfg["bar_minutes"])
    horizon = int(cfg["horizon"])
    vol_multiplier = float(cfg["vol_multiplier"])

    raw = load_deep(args.deep_root)
    schedule = deep_roll_schedule(raw)
    stitched = stitch_deep(raw, schedule)
    bars = bars_for_resolution(stitched, bar_minutes)
    matrix = build_matrix(bars, horizon, vol_multiplier)
    matrix["trade_week"] = trade_week_key(matrix["timestamp"])

    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    weekly_rows = []
    fit_receipts = []
    aggregate_cls = []

    min_week_rows = 3000 if bar_minutes == 1 else 40
    min_train_rows = 150000 if bar_minutes == 1 else 5000

    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        test_mask = (matrix["timestamp"] >= start) & (matrix["timestamp"] < end)
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < max(1000, min_week_rows):
            continue
        test_start_idx = int(test_idx[0])
        train_end = test_start_idx - horizon
        if train_end < min_train_rows:
            continue
        train = matrix.iloc[:train_end]
        test = matrix.iloc[int(test_idx[0]) : int(test_idx[-1] + 1)].copy()
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end)].copy()
        if len(train) < min_train_rows or len(test) < min_week_rows:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("walk-forward chronology violation")
        y_train = train["target"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 3:
            continue

        model = model_for(str(cfg["model"])).fit(train[FEATURES].to_numpy(float), y_train)
        pred = model.predict(test[FEATURES].to_numpy(float)).astype(int)
        y_test = test["target"].astype(int).to_numpy()
        cls = classification(y_test, pred)
        aggregate_cls.append({"quarter": f"{start.year}Q{((start.month - 1)//3)+1}", **cls})
        fit_receipts.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
        })
        test["model_pred"] = pred

        for week_key, week in test.groupby("trade_week", sort=True):
            if len(week) < min_week_rows:
                continue
            model_pred = week["model_pred"].to_numpy(int)
            point_move = week["point_move"].to_numpy(float)
            policies = {}
            for name, pp in {
                "model": model_pred,
                "always_long": np.ones(len(week), dtype=int),
                "always_short": -np.ones(len(week), dtype=int),
            }.items():
                policies[name] = {
                    "classification": classification(week["target"].astype(int).to_numpy(), pp),
                    "phase_audit": phase_audit(week["timestamp"], pp, point_move, bar_minutes, horizon),
                }
            weekly_rows.append({
                "trade_week": pd.Timestamp(week_key).isoformat(),
                "year": str(pd.Timestamp(week_key).year),
                "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
                "rows": int(len(week)),
                "policies": policies,
            })

    by_week = {}
    for row in weekly_rows:
        key = row["trade_week"]
        if key not in by_week or row["rows"] > by_week[key]["rows"]:
            by_week[key] = row
        elif row["rows"] == by_week[key]["rows"]:
            raise RuntimeError(f"ambiguous duplicate trade week {key}")
    rows = [by_week[k] for k in sorted(by_week)]
    if not rows:
        raise RuntimeError("no weekly OOS economics produced")

    summary = {
        policy: {f"after_{str(cost).replace('.', 'p')}pt": economic_summary(rows, policy, cost) for cost in POINT_COSTS}
        for policy in ("model", "always_long", "always_short")
    }
    support_1pt = summary["model"]["after_1p0pt"]["weeks"]
    foundation_status = "sufficient_weekly_support" if support_1pt >= 52 else "insufficient_weekly_support"

    result = {
        "schema": "foundry.mnq_multires_foundation.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "config_key": args.config_key,
        "bar_minutes": bar_minutes,
        "horizon_bars": horizon,
        "forecast_minutes": bar_minutes * horizon,
        "vol_multiplier": vol_multiplier,
        "model_contract": (
            "StandardScaler + class-balanced SGDClassifier(log_loss, average=True, random_state=42)"
            if cfg["model"] == "sgd"
            else "StandardScaler + class-balanced LogisticRegression(max_iter=2500, random_state=42)"
        ),
        "feature_contract": FEATURES,
        "target_contract": "three-class future log move versus +/- vol_multiplier * rv120 * sqrt(horizon)",
        "protocol": "quarterly expanding past-only walk-forward; horizon purge; same causal feature family per resolution; every UTC non-overlap phase and complete-enough week reported; no tuning or post-hoc phase/week/regime selection",
        "fit_receipts": fit_receipts,
        "quarter_classification": aggregate_cls,
        "weekly_rows": rows,
        "summary": summary,
        "model_after_1pt_by_year": decomposition(rows, "model", 1.0, "year"),
        "model_after_1pt_by_quarter": decomposition(rows, "model", 1.0, "quarter"),
        "foundation_status": foundation_status,
        "next_authority": "foundation evidence only; no tuning/promotion authority",
        "nq_boundary": "NQ not used as an input; reserved for later external/pre-MNQ validation",
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("MNQ_MULTIRES_FOUNDATION=PASS")
    print("CONFIG=" + args.config_key)
    print("FOUNDATION_STATUS=" + foundation_status)
    print(json.dumps(summary, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
