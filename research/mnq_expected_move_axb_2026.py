from __future__ import annotations

import argparse
import glob
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

HORIZON = 24
VOL_MULT = 0.5
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
OOF_MAGNITUDE_QUANTILE = 0.50
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
AXB_PIN = "60abd3fb6369c6ce0b6a4a65b0f2562fc96b1264"


def ridge_model():
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])


def load_axb_mnq(root: Path) -> pd.DataFrame:
    matches = glob.glob(str(root / "MNQ" / "MNQ_1min_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one AXB MNQ 1min file, got {matches}")
    f = pd.read_csv(matches[0])
    required = ["datetime", "open", "high", "low", "close", "volume"]
    if list(f.columns) != required:
        raise RuntimeError(f"unexpected AXB schema {list(f.columns)}")
    f["timestamp"] = pd.to_datetime(f["datetime"], utc=True, errors="raise")
    for c in required[1:]:
        f[c] = pd.to_numeric(f[c], errors="raise")
    f = f.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
    if not f["timestamp"].is_monotonic_increasing:
        raise RuntimeError("AXB timestamps not monotonic")
    w = f.set_index("timestamp")
    b = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), observed_minutes=("close", "count"),
    )
    b = b[b["observed_minutes"] > 0].reset_index()
    return b


def chronological_oof_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds = []
    folds = []
    for i in range(4):
        ts = first + i * fold
        te = n if i == 3 else first + (i + 1) * fold
        train_end = ts - HORIZON
        if train_end < 5000 or te - ts < 1000:
            raise RuntimeError(f"invalid OOF fold {i}")
        m = ridge_model()
        m.fit(x[:train_end], y[:train_end])
        preds.append(m.predict(x[ts:te]))
        folds.append({"fold": i, "train_rows": int(train_end), "test_rows": int(te-ts)})
    p = np.concatenate(preds)
    return float(np.quantile(np.abs(p), OOF_MAGNITUDE_QUANTILE)), {
        "oof_rows": int(len(p)),
        "abs_prediction_quantile": OOF_MAGNITUDE_QUANTILE,
        "threshold_abs_pred_z": float(np.quantile(np.abs(p), OOF_MAGNITUDE_QUANTILE)),
        "folds": folds,
    }


def summarize(rows: list[dict], policy: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        arr = np.asarray([
            r["policies"][policy]["phase_audit"].get(field)
            for r in rows
            if r["policies"][policy]["phase_audit"].get(field) is not None
        ], dtype=float)
        if len(arr) < 6:
            raise RuntimeError(f"insufficient independent-source weeks {policy}/{key}: {len(arr)}")
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
    out["coverage"] = {"median": float(np.median(cov)), "mean": float(np.mean(cov)), "min": float(np.min(cov)), "max": float(np.max(cov))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--axb-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    features = list(BASE_FEATURES)

    deep = _add_features(deep_bars(stitch_deep(load_deep(args.deep_root), deep_roll_schedule(load_deep(args.deep_root)))))
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    deep = deep[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    deep_cls, _, _ = target_columns(deep, HORIZON, VOL_MULT)
    deep["class_target"] = deep_cls
    deep["point_move"] = deep["close"].shift(-HORIZON) - deep["close"]
    deep_scale = deep["close"].astype(float) * deep["rv_120"].astype(float) * math.sqrt(HORIZON)
    deep["target_move_z"] = deep["point_move"] / deep_scale.replace(0, np.nan)

    pre = deep[(deep["timestamp"] < CUTOFF) & deep["class_target"].notna() & deep["target_move_z"].notna()].copy()
    train = pre.iloc[:-HORIZON].copy()
    if len(train) < 50000:
        raise RuntimeError(f"insufficient deep training rows {len(train)}")

    axb = _add_features(load_axb_mnq(args.axb_root))
    axb = axb[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    axb_cls, _, _ = target_columns(axb, HORIZON, VOL_MULT)
    axb["class_target"] = axb_cls
    axb["point_move"] = axb["close"].shift(-HORIZON) - axb["close"]
    axb_scale = axb["close"].astype(float) * axb["rv_120"].astype(float) * math.sqrt(HORIZON)
    axb["target_move_z"] = axb["point_move"] / axb_scale.replace(0, np.nan)
    axb["trade_week"] = trade_week_key(axb["timestamp"])
    test = axb[(axb["timestamp"] >= CUTOFF) & axb["class_target"].notna() & axb["target_move_z"].notna() & axb["point_move"].notna()].copy()
    if len(test) < 3000:
        raise RuntimeError(f"insufficient AXB test rows {len(test)}")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("chronology overlap")

    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_reg = train["target_move_z"].to_numpy(float)
    classifier = classifier_model().fit(x_train, train["class_target"].astype(int).to_numpy())
    cls_pred = classifier.predict(x_test).astype(int)
    threshold, oof = chronological_oof_threshold(x_train, y_reg)
    ridge = ridge_model(); ridge.fit(x_train, y_reg)
    pred_z = ridge.predict(x_test)
    ridge_pred = np.where(np.abs(pred_z) >= threshold, np.sign(pred_z), 0).astype(int)
    policies_full = {
        "classification_logistic": cls_pred,
        "ridge_expected_move": ridge_pred,
        "always_long": np.ones(len(test), dtype=int),
        "always_short": -np.ones(len(test), dtype=int),
    }

    test = test.copy(); test["_pos"] = np.arange(len(test))
    rows = []
    for week_key, g in test.groupby("trade_week", sort=True):
        if len(g) < 300:
            continue
        pos = g["_pos"].to_numpy(int); move = g["point_move"].to_numpy(float)
        policies = {}
        for name, full_pred in policies_full.items():
            pred = full_pred[pos]
            policies[name] = {
                "coverage": float(np.mean(pred != 0)),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "phase_audit": phase_audit(g["timestamp"], pred, move, HORIZON),
            }
        rows.append({"trade_week": pd.Timestamp(week_key).isoformat(), "rows": int(len(g)), "policies": policies})

    result = {
        "schema": "foundry.mnq_expected_move_axb_2026.v1",
        "research_only": True,
        "promotion_authority": False,
        "training_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176, strictly pre-2026",
        "test_source": f"axb0306/cme-futures-ohlc@{AXB_PIN} MNQ 1-minute",
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "vol_multiplier": VOL_MULT,
        "train_rows": int(len(train)),
        "train_last_timestamp": train["timestamp"].max().isoformat(),
        "test_rows": int(len(test)),
        "test_first_timestamp": test["timestamp"].min().isoformat(),
        "test_last_timestamp": test["timestamp"].max().isoformat(),
        "ridge_contract": "Ridge(alpha=10) on normalized future move; frozen 50th-percentile absolute-prediction gate learned from four chronological deep-MNQ pre-2026 OOF folds",
        "oof_threshold_receipt": oof,
        "weekly_rows": rows,
        "summary": {p: summarize(rows, p) for p in policies_full},
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_EXPECTED_MOVE_AXB_2026=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
