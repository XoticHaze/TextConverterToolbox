from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep

SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
TEST_START = pd.Timestamp("2022-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
HORIZONS = (4, 12)
FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "range_frac", "body_frac", "log_volume", "volume_z_24", "volume_z_120",
    "rv_12", "rv_60", "rv_120", "ema12_dist", "ema48_dist",
    "mom_24", "mom_120", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
TARGETS = ("long_mae_z", "short_mae_z", "future_rv_z")


def bars_1h(stitched: pd.DataFrame) -> pd.DataFrame:
    w = stitched.set_index("timestamp")
    bars = w.resample("60min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        observed_minutes=("close", "count"),
        source_contract=("symbol", "first"),
        source_contract_last=("symbol", "last"),
    )
    bars = bars[bars["observed_minutes"] > 0].copy()
    mixed = bars["source_contract"] != bars["source_contract_last"]
    bars = bars.loc[~mixed].drop(columns=["source_contract_last"]).reset_index()
    bars.attrs["dropped_roll_boundary_bars"] = int(mixed.sum())
    return bars


def _future_extreme(series: pd.Series, horizon: int, kind: str) -> pd.Series:
    shifted = pd.concat([series.shift(-i) for i in range(1, horizon + 1)], axis=1)
    if kind == "max":
        return shifted.max(axis=1, skipna=False)
    if kind == "min":
        return shifted.min(axis=1, skipna=False)
    raise ValueError(kind)


def _future_std(ret1: pd.Series, horizon: int) -> pd.Series:
    shifted = pd.concat([ret1.shift(-i) for i in range(1, horizon + 1)], axis=1)
    return shifted.std(axis=1, ddof=0, skipna=False)


def _segment_matrix(group: pd.DataFrame, horizon: int) -> pd.DataFrame:
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

    future_high = _future_extreme(high, horizon, "max")
    future_low = _future_extreme(low, horizon, "min")
    future_rv = _future_std(ret1, horizon)
    scale = close * g["rv_120"] * math.sqrt(horizon)
    g["long_mae_z"] = (close - future_low).clip(lower=0) / scale.replace(0, np.nan)
    g["short_mae_z"] = (future_high - close).clip(lower=0) / scale.replace(0, np.nan)
    g["future_rv_z"] = future_rv / g["rv_120"].replace(0, np.nan)
    return g


def build_matrix(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = bars.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    segment = work["source_contract"].ne(work["source_contract"].shift()).cumsum()
    out = pd.concat(
        [_segment_matrix(g, horizon) for _, g in work.groupby(segment, sort=False)],
        ignore_index=True,
    )
    keep = ["timestamp", "source_contract", *FEATURES, *TARGETS]
    out = out[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if out["timestamp"].duplicated().any() or not out["timestamp"].is_monotonic_increasing:
        raise RuntimeError("1H risk matrix timestamps are not unique/increasing")
    return out


def model_for(target: str) -> HistGradientBoostingRegressor:
    common = dict(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    if target in ("long_mae_z", "short_mae_z"):
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.80, **common)
    if target == "future_rv_z":
        return HistGradientBoostingRegressor(loss="squared_error", **common)
    raise ValueError(target)


def corr(pred: np.ndarray, truth: np.ndarray, method: str) -> float | None:
    p = pd.Series(np.asarray(pred, dtype=float))
    t = pd.Series(np.asarray(truth, dtype=float))
    mask = p.notna() & t.notna()
    if int(mask.sum()) < 100 or p[mask].nunique() < 2 or t[mask].nunique() < 2:
        return None
    return float(p[mask].corr(t[mask], method=method))


def train_cutpoints(train_pred: np.ndarray) -> list[float]:
    cuts = np.quantile(np.asarray(train_pred, dtype=float), [0.2, 0.4, 0.6, 0.8]).astype(float)
    if not np.all(np.diff(cuts) > 0):
        raise RuntimeError(f"non-unique training prediction quintile cuts: {cuts.tolist()}")
    return cuts.tolist()


def cohort_metrics(test_pred: np.ndarray, truth: np.ndarray, cuts: list[float]) -> dict:
    bucket = np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(test_pred, dtype=float), side="right")
    means = []
    counts = []
    for q in range(5):
        vals = np.asarray(truth, dtype=float)[bucket == q]
        counts.append(int(len(vals)))
        means.append(float(np.mean(vals)) if len(vals) else None)
    if any(v is None for v in means):
        q5_gt_q1 = False
        monotonic = False
        ratio = None
    else:
        q5_gt_q1 = bool(means[4] > means[0])
        monotonic = bool(np.all(np.diff(np.asarray(means, dtype=float)) >= 0))
        ratio = float(means[4] / means[0]) if means[0] > 0 else None
    return {
        "training_prediction_quintile_cuts": cuts,
        "test_counts": counts,
        "realized_means": means,
        "q5_gt_q1": q5_gt_q1,
        "monotonic_non_decreasing": monotonic,
        "q5_q1_ratio": ratio,
    }


def evaluate_horizon(matrix: pd.DataFrame, horizon: int) -> dict:
    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    folds = []
    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        test_mask = (matrix["timestamp"] >= start) & (matrix["timestamp"] < end)
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < 800:
            continue
        test_start_idx = int(test_idx[0])
        train_end = test_start_idx - horizon
        if train_end < 5000:
            continue
        train = matrix.iloc[:train_end]
        test = matrix.iloc[int(test_idx[0]) : int(test_idx[-1] + 1)].copy()
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end)].copy()
        if len(train) < 5000 or len(test) < 800:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("risk-state walk-forward chronology violation")

        metrics = {}
        x_train = train[FEATURES].to_numpy(float)
        x_test = test[FEATURES].to_numpy(float)
        for target in TARGETS:
            model = model_for(target).fit(x_train, train[target].to_numpy(float))
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            truth = test[target].to_numpy(float)
            cuts = train_cutpoints(train_pred)
            metrics[target] = {
                "pearson": corr(test_pred, truth, "pearson"),
                "spearman": corr(test_pred, truth, "spearman"),
                "predicted_mean": float(np.mean(test_pred)),
                "realized_mean": float(np.mean(truth)),
                "cohorts": cohort_metrics(test_pred, truth, cuts),
            }

        folds.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "metrics": metrics,
        })

    if len(folds) < 12:
        raise RuntimeError(f"insufficient 1H risk-state quarters for H{horizon}: {len(folds)}")
    return {"folds": folds, "summary": summarize(folds)}


def summarize(folds: list[dict]) -> dict:
    out = {}
    for target in TARGETS:
        spearman = np.asarray(
            [f["metrics"][target]["spearman"] for f in folds if f["metrics"][target]["spearman"] is not None],
            dtype=float,
        )
        pearson = np.asarray(
            [f["metrics"][target]["pearson"] for f in folds if f["metrics"][target]["pearson"] is not None],
            dtype=float,
        )
        q5 = [bool(f["metrics"][target]["cohorts"]["q5_gt_q1"]) for f in folds]
        mono = [bool(f["metrics"][target]["cohorts"]["monotonic_non_decreasing"]) for f in folds]
        ratios = [
            f["metrics"][target]["cohorts"]["q5_q1_ratio"]
            for f in folds
            if f["metrics"][target]["cohorts"]["q5_q1_ratio"] is not None
        ]
        out[target] = {
            "quarters": int(len(folds)),
            "spearman_positive_quarters": int(np.sum(spearman > 0)),
            "spearman_median": float(np.median(spearman)),
            "spearman_mean": float(np.mean(spearman)),
            "spearman_min": float(np.min(spearman)),
            "pearson_median": float(np.median(pearson)),
            "pearson_mean": float(np.mean(pearson)),
            "q5_gt_q1_quarters": int(np.sum(q5)),
            "monotonic_quarters": int(np.sum(mono)),
            "median_q5_q1_ratio": float(np.median(ratios)) if ratios else None,
        }
    return out


def decision(summary: dict, target: str) -> dict:
    row = summary[target]
    if target in ("long_mae_z", "short_mae_z"):
        checks = {
            "positive_spearman_10_quarters": row["spearman_positive_quarters"] >= 10,
            "median_spearman_at_least_0p15": row["spearman_median"] >= 0.15,
            "q5_gt_q1_10_quarters": row["q5_gt_q1_quarters"] >= 10,
        }
    else:
        checks = {
            "positive_spearman_10_quarters": row["spearman_positive_quarters"] >= 10,
            "median_spearman_at_least_0p20": row["spearman_median"] >= 0.20,
        }
    return {
        "action": "advance_to_separate_economic_consumer" if all(checks.values()) else "reject_as_slow_risk_state_candidate",
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    schedule = deep_roll_schedule(raw)
    stitched = stitch_deep(raw, schedule)
    bars = bars_1h(stitched)

    horizons = {}
    decisions = {}
    for horizon in HORIZONS:
        matrix = build_matrix(bars, horizon)
        result = evaluate_horizon(matrix, horizon)
        horizons[str(horizon)] = result
        decisions[str(horizon)] = {
            target: decision(result["summary"], target) for target in TARGETS
        }
        print("HORIZON", horizon, json.dumps(result["summary"], sort_keys=True))
        print("DECISION", horizon, json.dumps(decisions[str(horizon)], sort_keys=True))

    result = {
        "schema": "foundry.mnq_h1_risk_state.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "bar_minutes": 60,
        "horizons": horizons,
        "decisions": decisions,
        "feature_contract": FEATURES,
        "protocol": "quarterly expanding walk-forward 2022-2025 with full-horizon purge; 1H MNQ-only causal features; train-only prediction quintile cutpoints; no tuning, NQ input, or post-hoc target/horizon selection",
        "next_authority": "risk-state evidence only; any economic consumer requires a separately frozen contract",
        "nq_boundary": "NQ excluded from input and threshold fitting; reserved for later independent/pre-MNQ validation after an MNQ risk hypothesis passes",
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H1_RISK_STATE=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
