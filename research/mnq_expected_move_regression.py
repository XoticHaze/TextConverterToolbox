from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_model_family_weekly_challenge import trade_week_key
from research.mnq_opportunity_target_matrix import model as classifier_model, target_columns
from research.nq_to_mnq_execution_transfer import phase_audit

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
REGRESSORS = ("ridge", "hist_gradient_regressor")
POLICIES = ("classification_logistic", "ridge_expected_move", "hist_gradient_expected_move")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
OOF_MAGNITUDE_QUANTILE = 0.50
MIN_OOS_WEEKS = 80
MIN_VALID_POLICY_WEEKS = 60


def make_regressor(name: str):
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if name == "hist_gradient_regressor":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=42,
        )
    raise RuntimeError(name)


def chronological_oof_threshold(name: str, x: np.ndarray, y: np.ndarray, horizon: int) -> tuple[float, dict]:
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds = []
    truth = []
    receipts = []
    for i in range(4):
        ts = first + i * fold
        te = n if i == 3 else first + (i + 1) * fold
        train_end = ts - horizon
        if train_end < 5000 or te - ts < 1000:
            raise RuntimeError(f"invalid inner fold {i} train_end={train_end} test={te-ts}")
        m = make_regressor(name)
        m.fit(x[:train_end], y[:train_end])
        p = m.predict(x[ts:te])
        preds.append(p)
        truth.append(y[ts:te])
        receipts.append({"fold": i, "train_rows": int(train_end), "test_rows": int(te-ts)})
    p = np.concatenate(preds)
    t = np.concatenate(truth)
    threshold = float(np.quantile(np.abs(p), OOF_MAGNITUDE_QUANTILE))
    corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 and np.std(t) > 0 else float("nan")
    return threshold, {
        "oof_rows": int(len(p)),
        "abs_prediction_quantile": OOF_MAGNITUDE_QUANTILE,
        "threshold_abs_pred_z": threshold,
        "oof_prediction_target_correlation": corr,
        "folds": receipts,
    }


def summarize(rows: list[dict], policy: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        vals = [r["policies"][policy]["phase_audit"].get(field) for r in rows]
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if len(arr) < MIN_VALID_POLICY_WEEKS:
            raise RuntimeError(f"insufficient valid weeks {policy}/{key}: {len(arr)}")
        k = max(1, int(np.ceil(0.10 * len(arr))))
        worst = np.sort(arr)[:k]
        out[f"after_{key}pt"] = {
            "weeks": int(len(arr)),
            "total_oos_weeks": int(len(rows)),
            "valid_week_fraction": float(len(arr) / len(rows)),
            "positive_weeks": int(np.sum(arr > 0)),
            "positive_week_fraction": float(np.mean(arr > 0)),
            "median_weekly_points": float(np.median(arr)),
            "mean_weekly_points": float(np.mean(arr)),
            "p10_weekly_points": float(np.quantile(arr, 0.10)),
            "bottom10pct_mean_points": float(np.mean(worst)),
            "worst_week_points": float(np.min(arr)),
            "best_week_points": float(np.max(arr)),
        }
    cov = np.asarray([r["policies"][policy]["coverage"] for r in rows], dtype=float)
    out["coverage"] = {"median": float(np.median(cov)), "mean": float(np.mean(cov)), "min": float(np.min(cov)), "max": float(np.max(cov))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]
    features = list(BASE_FEATURES)

    raw = load_deep(args.deep_root)
    work = _add_features(deep_bars(stitch_deep(raw, deep_roll_schedule(raw))))
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    work = work[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    cls_target, _, _ = target_columns(work, horizon, vol_mult)
    work["class_target"] = cls_target
    work["point_move"] = work["close"].shift(-horizon) - work["close"]
    # Normalize the economic target by causal local volatility so one model can
    # span different MNQ price/volatility eras. The prediction is converted back
    # to points at the current row before economic scoring.
    scale_points = work["close"].astype(float) * work["rv_120"].astype(float) * math.sqrt(horizon)
    work["target_move_z"] = work["point_move"] / scale_points.replace(0, np.nan)
    work["scale_points"] = scale_points
    work["trade_week"] = trade_week_key(work["timestamp"])

    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    weekly_rows = []
    fit_receipts = []
    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["class_target"].notna() & work["target_move_z"].notna()
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 2000:
            continue
        test_start = int(idx[0])
        train_end = test_start - horizon
        if train_end < 50000:
            continue
        train = work.iloc[:train_end]
        train = train[train["class_target"].notna() & train["target_move_z"].notna()].copy()
        test = work.iloc[int(idx[0]):int(idx[-1] + 1)]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["class_target"].notna() & test["target_move_z"].notna()].copy()
        if len(train) < 50000 or len(test) < 2000 or train["timestamp"].max() >= test["timestamp"].min():
            continue

        x_train = train[features].to_numpy(float)
        x_test = test[features].to_numpy(float)
        y_reg = train["target_move_z"].to_numpy(float)
        cls = classifier_model().fit(x_train, train["class_target"].astype(int).to_numpy())
        quarter_preds = {"classification_logistic": cls.predict(x_test).astype(int)}
        reg_receipts = {}
        for name in REGRESSORS:
            threshold, oof = chronological_oof_threshold(name, x_train, y_reg, horizon)
            fitted = make_regressor(name)
            fitted.fit(x_train, y_reg)
            pred_z = fitted.predict(x_test)
            pred_points = pred_z * test["scale_points"].to_numpy(float)
            signal = np.where(np.abs(pred_z) >= threshold, np.sign(pred_z), 0).astype(int)
            quarter_preds[name + "_expected_move"] = signal
            reg_receipts[name] = {
                **oof,
                "test_predicted_point_move_mean": float(np.mean(pred_points)),
                "test_predicted_point_move_median": float(np.median(pred_points)),
                "test_signal_coverage": float(np.mean(signal != 0)),
            }
        fit_receipts.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "regressors": reg_receipts,
        })

        # Use exact future-row positions; each week is evaluated by every fixed UTC phase.
        test = test.copy()
        test["_pos"] = np.arange(len(test))
        for week_key, g in test.groupby("trade_week", sort=True):
            if len(g) < 300:
                continue
            pos = g["_pos"].to_numpy(int)
            move = g["point_move"].to_numpy(float)
            policies = {}
            for policy in POLICIES:
                pred = quarter_preds[policy][pos]
                policies[policy] = {
                    "coverage": float(np.mean(pred != 0)),
                    "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "phase_audit": phase_audit(g["timestamp"], pred, move, horizon),
                }
            weekly_rows.append({"trade_week": pd.Timestamp(week_key).isoformat(), "quarter": f"{start.year}Q{((start.month - 1)//3)+1}", "rows": int(len(g)), "policies": policies})

    by_week = {}
    for r in weekly_rows:
        k = r["trade_week"]
        if k not in by_week or r["rows"] > by_week[k]["rows"]:
            by_week[k] = r
    rows = [by_week[k] for k in sorted(by_week)]
    if len(rows) < MIN_OOS_WEEKS:
        raise RuntimeError(f"insufficient OOS weeks {len(rows)}")

    result = {
        "schema": "foundry.mnq_expected_move_regression.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier_for_classification_comparator": vol_mult,
        "feature_set": "baseline20",
        "economic_target": "future MNQ point move normalized by current close * rv_120 * sqrt(horizon); regress normalized move and convert prediction back to points for diagnostics",
        "selection_contract": "for each regression family, four chronological past-only inner OOF folds determine only the 50th percentile absolute predicted normalized-move magnitude; full training fit then trades the future quarter only when abs(prediction) clears that frozen magnitude threshold; no OOF PnL or future economics select the threshold",
        "policies": list(POLICIES),
        "fit_receipts": fit_receipts,
        "weekly_rows": rows,
        "summary": {p: summarize(rows, p) for p in POLICIES},
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_EXPECTED_MOVE_REGRESSION=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
