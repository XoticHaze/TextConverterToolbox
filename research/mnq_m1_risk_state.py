from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.mnq_external_transfer_validation import deep_roll_schedule, stitch_deep

SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
TEST_START = pd.Timestamp("2023-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
HORIZONS = (12, 24)
MIN_TRAIN_ROWS = 150_000
MIN_TEST_ROWS = 30_000
FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_30", "ret_60",
    "range_frac", "body_frac",
    "log_volume", "volume_z_30", "volume_z_120",
    "rv_30", "rv_120", "rv_390",
    "ema12_dist", "ema60_dist",
    "mom_30", "mom_120",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
TARGETS = ("long_mae_z", "short_mae_z", "future_rv_z")


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


def segment_matrix(group: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = group.copy().reset_index(drop=True)
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    open_ = g["open"].astype(float)
    log_close = np.log(close)
    ret1 = log_close.diff()

    for lag in (1, 3, 6, 12, 30, 60):
        g[f"ret_{lag}"] = log_close.diff(lag)
    g["range_frac"] = (high - low) / close.replace(0, np.nan)
    g["body_frac"] = (close - open_) / open_.replace(0, np.nan)
    g["log_volume"] = np.log1p(g["volume"].astype(float).clip(lower=0))
    for window in (30, 120):
        mean = g["log_volume"].rolling(window, min_periods=window).mean()
        std = g["log_volume"].rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
        g[f"volume_z_{window}"] = (g["log_volume"] - mean) / std
    for window in (30, 120, 390):
        g[f"rv_{window}"] = ret1.rolling(window, min_periods=window).std(ddof=0)
    for span in (12, 60):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        g[f"ema{span}_dist"] = close / ema - 1.0
    g["mom_30"] = close.pct_change(30)
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


def build_matrix(stitched: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = stitched[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    work = work.rename(columns={"symbol": "source_contract"})
    segment = work["source_contract"].ne(work["source_contract"].shift()).cumsum()
    parts = [segment_matrix(g, horizon) for _, g in work.groupby(segment, sort=False)]
    out = pd.concat(parts, ignore_index=True)
    keep = ["timestamp", "source_contract", *FEATURES, *TARGETS]
    out = out[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if out["timestamp"].duplicated().any() or not out["timestamp"].is_monotonic_increasing:
        raise RuntimeError("1Min risk-state matrix timestamps are not unique/increasing")
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
    means: list[float | None] = []
    counts: list[int] = []
    for q in range(5):
        vals = np.asarray(truth, dtype=float)[bucket == q]
        counts.append(int(len(vals)))
        means.append(float(np.mean(vals)) if len(vals) else None)
    if any(v is None for v in means):
        q5_gt_q1 = False
        monotonic = False
        ratio = None
    else:
        arr = np.asarray(means, dtype=float)
        q5_gt_q1 = bool(arr[4] > arr[0])
        monotonic = bool(np.all(np.diff(arr) >= 0))
        ratio = float(arr[4] / arr[0]) if arr[0] > 0 else None
    return {
        "training_prediction_quintile_cuts": cuts,
        "test_counts": counts,
        "realized_means": means,
        "q5_gt_q1": q5_gt_q1,
        "monotonic_non_decreasing": monotonic,
        "q5_q1_ratio": ratio,
    }


def summarize(folds: list[dict]) -> dict:
    out: dict[str, dict] = {}
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


def evaluate_horizon(matrix: pd.DataFrame, horizon: int) -> dict:
    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    folds: list[dict] = []
    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        test_mask = (matrix["timestamp"] >= start) & (matrix["timestamp"] < end)
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < MIN_TEST_ROWS:
            continue
        test_start_idx = int(test_idx[0])
        train_end = test_start_idx - horizon
        if train_end < MIN_TRAIN_ROWS:
            continue
        train = matrix.iloc[:train_end]
        test = matrix.iloc[int(test_idx[0]) : int(test_idx[-1] + 1)].copy()
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end)].copy()
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("1Min risk-state walk-forward chronology violation")

        metrics: dict[str, dict] = {}
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
            "quarter": f"{start.year}Q{((start.month - 1) // 3) + 1}",
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "metrics": metrics,
        })

    if len(folds) < 12:
        raise RuntimeError(f"insufficient 1Min risk-state quarters for H{horizon}: {len(folds)}")
    return {"folds": folds, "summary": summarize(folds)}


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
        "action": "advance_to_separate_fast_risk_consumer" if all(checks.values()) else "reject_as_fast_risk_state_candidate",
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))

    horizons: dict[str, dict] = {}
    decisions: dict[str, dict] = {}
    for horizon in HORIZONS:
        matrix = build_matrix(stitched, horizon)
        result = evaluate_horizon(matrix, horizon)
        horizons[str(horizon)] = result
        decisions[str(horizon)] = {
            target: decision(result["summary"], target) for target in TARGETS
        }
        print("HORIZON", horizon, json.dumps(result["summary"], sort_keys=True))
        print("DECISION", horizon, json.dumps(decisions[str(horizon)], sort_keys=True))

    result = {
        "schema": "foundry.mnq_m1_risk_state.v1",
        "research_only": True,
        "promotion_authority": False,
        "contract": "research/mnq_m1_risk_state_contract_20260901.json",
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "deep_timestamp_contract": contract_receipt(),
        "bar_minutes": 1,
        "horizons": horizons,
        "decisions": decisions,
        "feature_contract": FEATURES,
        "protocol": "quarterly expanding walk-forward 2023-2025 with full-horizon purge; corrected 1Min MNQ-only causal features; train-only prediction quintiles; all H12/H24 targets reported; no tuning or return-direction reinterpretation",
        "next_authority": "fast risk-state evidence only; any timing, stop, admission, sizing or 12Min consumer requires a separately frozen economic contract",
        "nq_boundary": "NQ excluded from model inputs and threshold fitting; reserved for later independent validation after a specific MNQ fast-risk consumer survives discovery",
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_M1_RISK_STATE=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
