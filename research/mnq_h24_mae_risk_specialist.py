from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_model_family_weekly_challenge import trade_week_key
from research.mnq_opportunity_target_matrix import model as classifier_model, target_columns
from research.nq_to_mnq_execution_transfer import phase_audit

HORIZON = 24
VOL_MULT = 0.5
TEST_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
RISK_QUANTILE = 0.80
OOF_VETO_QUANTILE = 0.75
MIN_OOS_WEEKS = 80
MIN_VALID_POLICY_WEEKS = 60
POLICIES = ("logistic", "mae_risk_veto")


def risk_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=RISK_QUANTILE,
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


def future_window_extreme(s: pd.Series, horizon: int, kind: str) -> pd.Series:
    shifted = pd.concat([s.shift(-i) for i in range(1, horizon + 1)], axis=1)
    if kind == "max":
        out = shifted.max(axis=1, skipna=False)
    elif kind == "min":
        out = shifted.min(axis=1, skipna=False)
    else:
        raise RuntimeError(kind)
    return out


def oof_risk_threshold(x: np.ndarray, y: np.ndarray, horizon: int) -> tuple[float, dict]:
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds = []
    truth = []
    folds = []
    for i in range(4):
        ts = first + i * fold
        te = n if i == 3 else first + (i + 1) * fold
        train_end = ts - horizon
        if train_end < 5000 or te - ts < 1000:
            raise RuntimeError(f"invalid risk inner fold {i}: train_end={train_end} test={te-ts}")
        m = risk_model()
        m.fit(x[:train_end], y[:train_end])
        p = m.predict(x[ts:te])
        preds.append(p)
        truth.append(y[ts:te])
        folds.append({"fold": i, "train_rows": int(train_end), "test_rows": int(te-ts)})
    p = np.concatenate(preds)
    t = np.concatenate(truth)
    threshold = float(np.quantile(p, OOF_VETO_QUANTILE))
    corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 and np.std(t) > 0 else float("nan")
    return threshold, {
        "oof_rows": int(len(p)),
        "risk_prediction_quantile": RISK_QUANTILE,
        "veto_quantile_of_oof_predicted_risk": OOF_VETO_QUANTILE,
        "threshold_predicted_mae_z": threshold,
        "oof_prediction_realized_mae_correlation": corr,
        "folds": folds,
    }


def tail_summary(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < MIN_VALID_POLICY_WEEKS:
        raise RuntimeError(f"insufficient valid policy weeks {len(vals)}")
    k = max(1, int(np.ceil(0.10 * len(vals))))
    worst = np.sort(vals)[:k]
    return {
        "weeks": int(len(vals)),
        "positive_weeks": int(np.sum(vals > 0)),
        "positive_week_fraction": float(np.mean(vals > 0)),
        "median_weekly_points": float(np.median(vals)),
        "mean_weekly_points": float(np.mean(vals)),
        "p10_weekly_points": float(np.quantile(vals, 0.10)),
        "bottom10pct_mean_points": float(np.mean(worst)),
        "worst_week_points": float(np.min(vals)),
        "best_week_points": float(np.max(vals)),
    }


def summarize(rows: list[dict], policy: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        vals = [r["policies"][policy]["phase_audit"].get(field) for r in rows]
        out[f"after_{key}pt"] = tail_summary(np.asarray([v for v in vals if v is not None], dtype=float))
    cov = np.asarray([r["policies"][policy]["coverage"] for r in rows], dtype=float)
    out["coverage"] = {
        "median": float(np.median(cov)),
        "mean": float(np.mean(cov)),
        "min": float(np.min(cov)),
        "max": float(np.max(cov)),
    }
    return out


def paired(rows: list[dict]) -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        vals = []
        for r in rows:
            a = r["policies"]["mae_risk_veto"]["phase_audit"].get(field)
            b = r["policies"]["logistic"]["phase_audit"].get(field)
            if a is not None and b is not None:
                vals.append(float(a) - float(b))
        arr = np.asarray(vals, dtype=float)
        if len(arr) < MIN_VALID_POLICY_WEEKS:
            raise RuntimeError(f"insufficient paired weeks {len(arr)}")
        out[f"after_{key}pt"] = {
            "weeks": int(len(arr)),
            "median_delta_points": float(np.median(arr)),
            "mean_delta_points": float(np.mean(arr)),
            "win_fraction": float(np.mean(arr > 0)),
            "loss_fraction": float(np.mean(arr < 0)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    features = list(BASE_FEATURES)
    raw = load_deep(args.deep_root)
    bars = deep_bars(stitch_deep(raw, deep_roll_schedule(raw)))
    work = _add_features(bars)
    needed = list(dict.fromkeys(["timestamp", "open", "high", "low", "close", "rv_120", *features]))
    work = work[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    cls, _, _ = target_columns(work, HORIZON, VOL_MULT)
    work["class_target"] = cls
    work["point_move"] = work["close"].shift(-HORIZON) - work["close"]
    future_high = future_window_extreme(work["high"].astype(float), HORIZON, "max")
    future_low = future_window_extreme(work["low"].astype(float), HORIZON, "min")
    scale = work["close"].astype(float) * work["rv_120"].astype(float) * math.sqrt(HORIZON)
    work["long_mae_z"] = (work["close"] - future_low).clip(lower=0) / scale.replace(0, np.nan)
    work["short_mae_z"] = (future_high - work["close"]).clip(lower=0) / scale.replace(0, np.nan)
    work["trade_week"] = trade_week_key(work["timestamp"])

    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    weekly_rows = []
    fit_receipts = []
    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        mask = (
            (work["timestamp"] >= start) & (work["timestamp"] < end)
            & work["class_target"].notna() & work["point_move"].notna()
            & work["long_mae_z"].notna() & work["short_mae_z"].notna()
        )
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 2000:
            continue
        test_start = int(idx[0])
        train_end = test_start - HORIZON
        if train_end < 50000:
            continue
        train = work.iloc[:train_end]
        train = train[
            train["class_target"].notna() & train["long_mae_z"].notna() & train["short_mae_z"].notna()
        ].copy()
        test = work.iloc[int(idx[0]):int(idx[-1] + 1)]
        test = test[
            (test["timestamp"] >= start) & (test["timestamp"] < end)
            & test["class_target"].notna() & test["point_move"].notna()
            & test["long_mae_z"].notna() & test["short_mae_z"].notna()
        ].copy()
        if len(train) < 50000 or len(test) < 2000 or train["timestamp"].max() >= test["timestamp"].min():
            continue

        x_train = train[features].to_numpy(float)
        x_test = test[features].to_numpy(float)
        anchor = classifier_model().fit(x_train, train["class_target"].astype(int).to_numpy())
        anchor_pred = anchor.predict(x_test).astype(int)

        risk_predictions = {}
        risk_receipts = {}
        for side, target in (("long", "long_mae_z"), ("short", "short_mae_z")):
            y = train[target].to_numpy(float)
            threshold, oof = oof_risk_threshold(x_train, y, HORIZON)
            m = risk_model()
            m.fit(x_train, y)
            pred = m.predict(x_test)
            risk_predictions[side] = pred
            realized = test[target].to_numpy(float)
            corr = float(np.corrcoef(pred, realized)[0, 1]) if np.std(pred) > 0 and np.std(realized) > 0 else float("nan")
            risk_receipts[side] = {
                **oof,
                "test_prediction_realized_mae_correlation": corr,
                "test_predicted_mae_z_mean": float(np.mean(pred)),
                "test_predicted_mae_z_median": float(np.median(pred)),
            }

        risk_for_signal = np.full(len(test), np.nan, dtype=float)
        threshold_for_signal = np.full(len(test), np.nan, dtype=float)
        long_mask = anchor_pred == 1
        short_mask = anchor_pred == -1
        risk_for_signal[long_mask] = risk_predictions["long"][long_mask]
        risk_for_signal[short_mask] = risk_predictions["short"][short_mask]
        threshold_for_signal[long_mask] = risk_receipts["long"]["threshold_predicted_mae_z"]
        threshold_for_signal[short_mask] = risk_receipts["short"]["threshold_predicted_mae_z"]
        veto = (anchor_pred != 0) & np.isfinite(risk_for_signal) & (risk_for_signal >= threshold_for_signal)
        risk_veto_pred = anchor_pred.copy()
        risk_veto_pred[veto] = 0

        fit_receipts.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "risk_models": risk_receipts,
            "anchor_signal_coverage": float(np.mean(anchor_pred != 0)),
            "vetoed_anchor_fraction": float(np.mean(veto[anchor_pred != 0])) if np.any(anchor_pred != 0) else 0.0,
        })

        test = test.copy()
        test["_pos"] = np.arange(len(test))
        preds = {"logistic": anchor_pred, "mae_risk_veto": risk_veto_pred}
        for week_key, g in test.groupby("trade_week", sort=True):
            if len(g) < 300:
                continue
            pos = g["_pos"].to_numpy(int)
            move = g["point_move"].to_numpy(float)
            policies = {}
            for policy in POLICIES:
                pred = preds[policy][pos]
                policies[policy] = {
                    "coverage": float(np.mean(pred != 0)),
                    "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "phase_audit": phase_audit(g["timestamp"], pred, move, HORIZON),
                }
            weekly_rows.append({
                "trade_week": pd.Timestamp(week_key).isoformat(),
                "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
                "rows": int(len(g)),
                "policies": policies,
            })

    by_week = {}
    for r in weekly_rows:
        k = r["trade_week"]
        if k not in by_week or r["rows"] > by_week[k]["rows"]:
            by_week[k] = r
    rows = [by_week[k] for k in sorted(by_week)]
    if len(rows) < MIN_OOS_WEEKS:
        raise RuntimeError(f"insufficient OOS weeks {len(rows)}")

    result = {
        "schema": "foundry.mnq_h24_mae_risk_specialist.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "vol_multiplier": VOL_MULT,
        "risk_target": "direction-specific future maximum adverse excursion over next H24 bars, normalized by current close*rv_120*sqrt(H); long and short risk models are separate",
        "risk_model": "HistGradientBoostingRegressor quantile=0.80 with fixed parameters",
        "veto_contract": "quarterly past-only anchor fit; four chronological inner OOF risk folds define only the 75th percentile of predicted side-specific MAE risk; future anchor trades are vetoed only when their side-specific predicted risk is at/above that frozen OOF threshold; no OOF/future PnL tunes the threshold",
        "policies": list(POLICIES),
        "fit_receipts": fit_receipts,
        "weekly_rows": rows,
        "summary": {p: summarize(rows, p) for p in POLICIES},
        "paired_mae_veto_minus_logistic": paired(rows),
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_MAE_RISK_SPECIALIST=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print(json.dumps(result["paired_mae_veto_minus_logistic"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
