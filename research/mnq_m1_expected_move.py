from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.mnq_external_transfer_validation import deep_roll_schedule, stitch_deep

SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
DISCOVERY_START = pd.Timestamp("2023-07-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2026-01-01", tz="UTC")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
CONFIGS = {
    "m1_h30": {"horizon": 30, "vol_multiplier": 0.5},
    "m1_h60": {"horizon": 60, "vol_multiplier": 0.5},
}
FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_30", "ret_60",
    "range_frac", "body_frac",
    "log_volume", "volume_z_30", "volume_z_120",
    "rv_30", "rv_120", "rv_390",
    "ema12_dist", "ema60_dist",
    "mom_30", "mom_120",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
MIN_TRAIN_ROWS = 150_000
MIN_WEEK_ROWS = 3_000
MIN_PAIRED_WEEKS = 80


def segment_features(group: pd.DataFrame, horizon: int, vol_multiplier: float) -> pd.DataFrame:
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

    future_close = close.shift(-horizon)
    point_move = future_close - close
    scale_points = close * g["rv_120"] * math.sqrt(horizon)
    g["point_move"] = point_move
    g["target_move_z"] = point_move / scale_points.replace(0, np.nan)

    fwd_log = np.log(future_close / close)
    class_threshold = vol_multiplier * g["rv_120"] * math.sqrt(horizon)
    valid = future_close.notna() & class_threshold.notna()
    target = pd.Series(np.nan, index=g.index, dtype=float)
    target.loc[valid] = 0.0
    target.loc[valid & (fwd_log > class_threshold)] = 1.0
    target.loc[valid & (fwd_log < -class_threshold)] = -1.0
    g["class_target"] = target
    return g


def build_matrix(stitched: pd.DataFrame, horizon: int, vol_multiplier: float) -> pd.DataFrame:
    work = stitched[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    work = work.rename(columns={"symbol": "source_contract"})
    segment = work["source_contract"].ne(work["source_contract"].shift()).cumsum()
    parts = [segment_features(g, horizon, vol_multiplier) for _, g in work.groupby(segment, sort=False)]
    out = pd.concat(parts, ignore_index=True)
    keep = ["timestamp", "source_contract", "close", *FEATURES, "point_move", "target_move_z", "class_target"]
    out = out[keep].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[*FEATURES, "point_move", "target_move_z", "class_target"]).reset_index(drop=True)
    out["class_target"] = out["class_target"].astype(int)
    if out["timestamp"].duplicated().any() or not out["timestamp"].is_monotonic_increasing:
        raise RuntimeError("1Min expected-move matrix timestamps are not unique/increasing")
    return out


def ridge_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0, solver="lsqr")),
    ])


def classifier_model() -> Pipeline:
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


def chronological_oof_threshold(x: np.ndarray, y: np.ndarray, horizon: int) -> tuple[float, dict]:
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds = []
    receipts = []
    for i in range(4):
        test_start = first + i * fold
        test_end = n if i == 3 else first + (i + 1) * fold
        train_end = test_start - horizon
        if train_end < MIN_TRAIN_ROWS or test_end - test_start < 20_000:
            raise RuntimeError(f"invalid 1Min inner OOF fold {i}: train={train_end} test={test_end-test_start}")
        model = ridge_model().fit(x[:train_end], y[:train_end])
        pred = model.predict(x[test_start:test_end])
        preds.append(np.asarray(pred, dtype=float))
        receipts.append({"fold": i, "train_rows": int(train_end), "test_rows": int(test_end - test_start)})
    all_pred = np.concatenate(preds)
    threshold = float(np.quantile(np.abs(all_pred), 0.50))
    return threshold, {
        "oof_rows": int(len(all_pred)),
        "abs_prediction_quantile": 0.50,
        "threshold_abs_pred_z": threshold,
        "folds": receipts,
    }


def trade_week_key(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    local = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    trade_date = local.dt.normalize() + pd.to_timedelta((local.dt.hour >= 18).astype(int), unit="D")
    return trade_date - pd.to_timedelta(trade_date.dt.weekday, unit="D")


def phase_audit(ts: pd.Series, signal: np.ndarray, point_move: np.ndarray, horizon: int) -> dict:
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    slots = ((parsed - epoch) // pd.Timedelta(minutes=1)).to_numpy(dtype=np.int64)
    phases = {}
    for phase in range(horizon):
        mask = (slots % horizon) == phase
        sig = signal[mask]
        move = point_move[mask]
        selected = sig != 0
        side = np.where(sig[selected] == 1, 1.0, -1.0)
        gross = side * move[selected]
        rec = {"signals": int(len(gross))}
        if len(gross) >= 10:
            rec["gross_mean_points"] = float(np.mean(gross))
            for cost in POINT_COSTS:
                key = str(cost).replace(".", "p")
                rec[f"net_mean_points_after_{key}pt"] = float(np.mean(gross - cost))
        phases[str(phase)] = rec
    valid = [v for v in phases.values() if "net_mean_points_after_1p0pt" in v]
    out = {
        "valid_phases": int(len(valid)),
        "phase_streams": phases,
        "contract": f"absolute UTC one-minute slot modulo H{horizon}; every phase reported; no post-hoc phase selection",
    }
    if valid:
        for cost in POINT_COSTS:
            key = str(cost).replace(".", "p")
            arr = np.asarray([float(v[f"net_mean_points_after_{key}pt"]) for v in valid], dtype=float)
            out[f"median_phase_net_points_after_{key}pt"] = float(np.median(arr))
            out[f"mean_phase_net_points_after_{key}pt"] = float(np.mean(arr))
            out[f"positive_phase_fraction_after_{key}pt"] = float(np.mean(arr > 0))
    return out


def policy_tail(rows: list[dict], policy: str, cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    values = []
    for row in rows:
        value = row["policies"][policy]["phase_audit"].get(field)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"weeks": 0}
    k = max(1, int(np.ceil(0.10 * len(arr))))
    cumulative = np.cumsum(arr)
    curve = np.r_[0.0, cumulative]
    peak = np.maximum.accumulate(curve)
    draw = curve - peak
    return {
        "weeks": int(len(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "positive_week_fraction": float(np.mean(arr > 0)),
        "p10": float(np.quantile(arr, 0.10)),
        "bottom10pct_mean": float(np.mean(np.sort(arr)[:k])),
        "worst_week": float(np.min(arr)),
        "best_week": float(np.max(arr)),
        "cumulative_points": float(np.sum(arr)),
        "max_drawdown_points": float(np.min(draw)),
    }


def paired_tail(rows: list[dict], challenger: str, reference: str, cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    a, b, labels = [], [], []
    for row in rows:
        av = row["policies"][challenger]["phase_audit"].get(field)
        bv = row["policies"][reference]["phase_audit"].get(field)
        if av is not None and bv is not None and np.isfinite(av) and np.isfinite(bv):
            a.append(float(av)); b.append(float(bv)); labels.append(row["trade_week"])
    aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float); delta = aa - bb
    if len(delta) == 0:
        return {"weeks": 0}
    return {
        "weeks": int(len(delta)),
        "median_delta": float(np.median(delta)),
        "mean_delta": float(np.mean(delta)),
        "win_fraction": float(np.mean(delta > 0)),
        "tie_fraction": float(np.mean(delta == 0)),
        "p10_delta": float(np.quantile(delta, 0.10)),
        "worst_delta": float(np.min(delta)),
        "best_delta": float(np.max(delta)),
        "first_week": labels[0],
        "last_week": labels[-1],
    }


def decomposition(rows: list[dict], policy: str, cost: float, group_field: str) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = row["policies"][policy]["phase_audit"].get(field)
        if value is not None and np.isfinite(value):
            groups.setdefault(str(row[group_field]), []).append(float(value))
    return {
        label: {
            "weeks": int(len(vals)),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "positive_fraction": float(np.mean(np.asarray(vals, dtype=float) > 0)),
        }
        for label, vals in sorted(groups.items())
    }


def advance_checks(rows: list[dict], cost: float) -> dict:
    candidate = policy_tail(rows, "ridge_expected_move", cost)
    control = policy_tail(rows, "direct_classifier", cost)
    paired = paired_tail(rows, "ridge_expected_move", "direct_classifier", cost)
    checks = {
        "paired_weeks_at_least_80": paired.get("weeks", 0) >= MIN_PAIRED_WEEKS,
        "candidate_median_positive": candidate.get("median", float("-inf")) > 0,
        "candidate_mean_positive": candidate.get("mean", float("-inf")) > 0,
        "candidate_positive_week_fraction_gt_0p50": candidate.get("positive_week_fraction", 0) > 0.50,
        "paired_median_delta_positive": paired.get("median_delta", float("-inf")) > 0,
        "paired_mean_delta_positive": paired.get("mean_delta", float("-inf")) > 0,
        "paired_win_fraction_gt_0p50": paired.get("win_fraction", 0) > 0.50,
        "candidate_p10_no_worse": candidate.get("p10", float("-inf")) >= control.get("p10", float("inf")),
        "candidate_bottom10pct_no_worse": candidate.get("bottom10pct_mean", float("-inf")) >= control.get("bottom10pct_mean", float("inf")),
        "candidate_drawdown_no_worse": candidate.get("max_drawdown_points", float("-inf")) >= control.get("max_drawdown_points", float("inf")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = CONFIGS[args.config_key]
    horizon = int(cfg["horizon"])
    vol_multiplier = float(cfg["vol_multiplier"])

    raw = load_deep(args.deep_root)
    schedule = deep_roll_schedule(raw)
    stitched = stitch_deep(raw, schedule)
    matrix = build_matrix(stitched, horizon, vol_multiplier)
    matrix["trade_week"] = trade_week_key(matrix["timestamp"])

    quarter_starts = list(pd.date_range(DISCOVERY_START, DISCOVERY_END, freq="QS", tz="UTC"))
    weekly_rows = []
    fit_receipts = []
    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        test_mask = (matrix["timestamp"] >= start) & (matrix["timestamp"] < end)
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < MIN_WEEK_ROWS:
            continue
        first_test = int(test_idx[0])
        train_end = first_test - horizon
        if train_end < MIN_TRAIN_ROWS:
            continue
        train = matrix.iloc[:train_end].copy()
        test = matrix.iloc[int(test_idx[0]): int(test_idx[-1] + 1)].copy()
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end)].copy()
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("1Min expected-move outer chronology violation")

        x_train = train[FEATURES].to_numpy(float)
        x_test = test[FEATURES].to_numpy(float)
        threshold, oof = chronological_oof_threshold(x_train, train["target_move_z"].to_numpy(float), horizon)
        ridge = ridge_model().fit(x_train, train["target_move_z"].to_numpy(float))
        pred_z = ridge.predict(x_test)
        ridge_signal = np.where(np.abs(pred_z) >= threshold, np.sign(pred_z), 0).astype(int)

        classifier = classifier_model().fit(x_train, train["class_target"].to_numpy(int))
        class_signal = classifier.predict(x_test).astype(int)
        class_truth = test["class_target"].to_numpy(int)

        fit_receipts.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "ridge_oof": oof,
            "ridge_coverage": float(np.mean(ridge_signal != 0)),
            "ridge_long_signals": int(np.sum(ridge_signal == 1)),
            "ridge_short_signals": int(np.sum(ridge_signal == -1)),
            "direct_classifier": {
                "balanced_accuracy": float(balanced_accuracy_score(class_truth, class_signal)),
                "macro_f1": float(f1_score(class_truth, class_signal, average="macro", zero_division=0)),
                "coverage": float(np.mean(class_signal != 0)),
                "predicted_counts": {str(c): int(np.sum(class_signal == c)) for c in (-1, 0, 1)},
            },
        })

        test = test.copy()
        test["_pos"] = np.arange(len(test), dtype=int)
        test["quarter"] = f"{start.year}Q{((start.month - 1)//3)+1}"
        for week_key, group in test.groupby("trade_week", sort=True):
            if len(group) < MIN_WEEK_ROWS:
                continue
            pos = group["_pos"].to_numpy(int)
            move = group["point_move"].to_numpy(float)
            policies = {
                "ridge_expected_move": ridge_signal[pos],
                "direct_classifier": class_signal[pos],
                "always_long": np.ones(len(group), dtype=int),
                "always_short": -np.ones(len(group), dtype=int),
                "flat": np.zeros(len(group), dtype=int),
            }
            weekly_rows.append({
                "trade_week": pd.Timestamp(week_key).isoformat(),
                "year": str(pd.Timestamp(week_key).year),
                "quarter": group["quarter"].iloc[0],
                "rows": int(len(group)),
                "policies": {
                    name: {
                        "coverage": float(np.mean(sig != 0)),
                        "signal_counts": {str(c): int(np.sum(sig == c)) for c in (-1, 0, 1)},
                        "phase_audit": phase_audit(group["timestamp"], sig, move, horizon),
                    }
                    for name, sig in policies.items()
                },
            })

    by_week: dict[str, dict] = {}
    for row in weekly_rows:
        key = row["trade_week"]
        if key not in by_week or row["rows"] > by_week[key]["rows"]:
            by_week[key] = row
    rows = [by_week[key] for key in sorted(by_week)]

    summaries = {
        policy: {str(cost): policy_tail(rows, policy, cost) for cost in POINT_COSTS}
        for policy in ("ridge_expected_move", "direct_classifier", "always_long", "always_short")
    }
    paired = {str(cost): paired_tail(rows, "ridge_expected_move", "direct_classifier", cost) for cost in POINT_COSTS}
    checks = {"1.0": advance_checks(rows, 1.0), "2.0": advance_checks(rows, 2.0)}
    passed = checks["1.0"]["passed"] and checks["2.0"]["passed"]

    result = {
        "schema": "foundry.mnq_m1_expected_move.v1",
        "research_only": True,
        "promotion_authority": False,
        "contract": "research/mnq_m1_expected_move_contract_20260901.json",
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "deep_timestamp_contract": contract_receipt(),
        "config_key": args.config_key,
        "horizon": horizon,
        "classifier_vol_multiplier": vol_multiplier,
        "features": FEATURES,
        "discovery_period": {"start": DISCOVERY_START.isoformat(), "end_exclusive": DISCOVERY_END.isoformat()},
        "fit_receipts": fit_receipts,
        "weekly_rows": rows,
        "summary": summaries,
        "paired_expected_move_minus_direct_classifier": paired,
        "advance_checks": checks,
        "decision": "advance_unchanged_to_corrected_2026_confirmation" if passed else "reject_as_specified",
        "decomposition_after_1pt": {
            "ridge_year": decomposition(rows, "ridge_expected_move", 1.0, "year"),
            "ridge_quarter": decomposition(rows, "ridge_expected_move", 1.0, "quarter"),
            "direct_year": decomposition(rows, "direct_classifier", 1.0, "year"),
            "direct_quarter": decomposition(rows, "direct_classifier", 1.0, "quarter"),
        },
        "nq_boundary": "NQ excluded from inputs and threshold fitting; later independent validation only after unchanged MNQ discovery and corrected-2026 pass.",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_M1_EXPECTED_MOVE=PASS")
    print("CONFIG=" + args.config_key)
    print("DECISION=" + result["decision"])
    print("CHECKS=" + json.dumps(checks, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
