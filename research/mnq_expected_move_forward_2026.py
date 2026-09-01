from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_model_family_weekly_challenge import trade_week_key
from research.mnq_opportunity_target_matrix import model as classifier_model, target_columns
from research.nq_to_mnq_execution_transfer import phase_audit

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
OOF_MAGNITUDE_QUANTILE = 0.50
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)


def ridge_model():
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])


def chronological_oof_threshold(x: np.ndarray, y: np.ndarray, horizon: int) -> tuple[float, dict]:
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds = []
    folds = []
    for i in range(4):
        ts = first + i * fold
        te = n if i == 3 else first + (i + 1) * fold
        train_end = ts - horizon
        if train_end < 5000 or te - ts < 1000:
            raise RuntimeError(f"invalid OOF fold {i}: train_end={train_end} test={te-ts}")
        m = ridge_model()
        m.fit(x[:train_end], y[:train_end])
        preds.append(m.predict(x[ts:te]))
        folds.append({"fold": i, "train_rows": int(train_end), "test_rows": int(te-ts)})
    p = np.concatenate(preds)
    threshold = float(np.quantile(np.abs(p), OOF_MAGNITUDE_QUANTILE))
    return threshold, {
        "oof_rows": int(len(p)),
        "abs_prediction_quantile": OOF_MAGNITUDE_QUANTILE,
        "threshold_abs_pred_z": threshold,
        "folds": folds,
    }


def signed_points(pred: np.ndarray, point_move: np.ndarray) -> np.ndarray:
    mask = pred != 0
    if not mask.any():
        return np.asarray([], dtype=float)
    side = np.where(pred[mask] == 1, 1.0, -1.0)
    return side * point_move[mask]


def summarize_weeks(rows: list[dict], policy: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        vals = [r["policies"][policy]["phase_audit"].get(field) for r in rows]
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if len(arr) < 6:
            raise RuntimeError(f"insufficient forward weeks {policy}/{key}: {len(arr)}")
        k = max(1, int(np.ceil(0.10 * len(arr))))
        worst = np.sort(arr)[:k]
        out[f"after_{key}pt"] = {
            "weeks": int(len(arr)),
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
    out["coverage"] = {
        "median": float(np.median(cov)),
        "mean": float(np.mean(cov)),
        "min": float(np.min(cov)),
        "max": float(np.max(cov)),
    }
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
    scale_points = work["close"].astype(float) * work["rv_120"].astype(float) * math.sqrt(horizon)
    work["target_move_z"] = work["point_move"] / scale_points.replace(0, np.nan)
    work["scale_points"] = scale_points
    work["trade_week"] = trade_week_key(work["timestamp"])

    pre = work[(work["timestamp"] < CUTOFF) & work["class_target"].notna() & work["target_move_z"].notna()].copy()
    if len(pre) <= horizon:
        raise RuntimeError("insufficient pre-2026 data")
    train = pre.iloc[:-horizon].copy()
    test = work[(work["timestamp"] >= CUTOFF) & work["class_target"].notna() & work["target_move_z"].notna() & work["point_move"].notna()].copy()
    if len(train) < 50000 or len(test) < 3000:
        raise RuntimeError(f"insufficient rows train={len(train)} test={len(test)}")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("chronology overlap")

    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_reg = train["target_move_z"].to_numpy(float)

    classifier = classifier_model().fit(x_train, train["class_target"].astype(int).to_numpy())
    cls_pred = classifier.predict(x_test).astype(int)

    threshold, oof = chronological_oof_threshold(x_train, y_reg, horizon)
    ridge = ridge_model()
    ridge.fit(x_train, y_reg)
    pred_z = ridge.predict(x_test)
    ridge_pred = np.where(np.abs(pred_z) >= threshold, np.sign(pred_z), 0).astype(int)

    policies_full = {
        "classification_logistic": cls_pred,
        "ridge_expected_move": ridge_pred,
        "always_long": np.ones(len(test), dtype=int),
        "always_short": -np.ones(len(test), dtype=int),
    }

    test = test.copy()
    test["_pos"] = np.arange(len(test))
    weekly_rows = []
    for week_key, g in test.groupby("trade_week", sort=True):
        if len(g) < 300:
            continue
        pos = g["_pos"].to_numpy(int)
        move = g["point_move"].to_numpy(float)
        policies = {}
        for name, full_pred in policies_full.items():
            pred = full_pred[pos]
            policies[name] = {
                "coverage": float(np.mean(pred != 0)),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "phase_audit": phase_audit(g["timestamp"], pred, move, horizon),
            }
        weekly_rows.append({
            "trade_week": pd.Timestamp(week_key).isoformat(),
            "rows": int(len(g)),
            "policies": policies,
        })

    policies = tuple(policies_full)
    result = {
        "schema": "foundry.mnq_expected_move_forward_2026.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier_for_classification_comparator": vol_mult,
        "feature_set": "baseline20",
        "train_cutoff": CUTOFF.isoformat(),
        "train_rows": int(len(train)),
        "train_last_timestamp": train["timestamp"].max().isoformat(),
        "test_rows": int(len(test)),
        "test_first_timestamp": test["timestamp"].min().isoformat(),
        "test_last_timestamp": test["timestamp"].max().isoformat(),
        "ridge_contract": "Ridge(alpha=10) on normalized future MNQ point move; fixed 50th-percentile absolute prediction gate learned only from four chronological pre-2026 OOF folds",
        "oof_threshold_receipt": oof,
        "weekly_rows": weekly_rows,
        "summary": {p: summarize_weeks(weekly_rows, p) for p in policies},
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_EXPECTED_MOVE_FORWARD_2026=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
