from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.expanded_regime_ablation import BASE_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_model_family_weekly_challenge import trade_week_key
from research.mnq_opportunity_target_matrix import classification, target_columns
from research.nq_to_mnq_execution_transfer import phase_audit

HORIZON = 12
VOL_MULTIPLIER = 1.0
TEST_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
MIN_COMPLETE_WEEKS = 80
INNER_FOLDS = 4
TRUST_QUANTILE = 0.65
MIN_META_ROWS = 12000
SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"

LOCAL_STATE_FEATURES = [
    "vol_ratio_12_120",
    "atr14_pct",
    "atr28_pct",
    "z_volume_120",
    "z_close_20",
    "z_close_60",
    "ema20_50_spread",
    "ema50_200_spread",
    "efficiency_30",
    "ret_skew_60",
    "ret_autocorr_60",
    "bb20_width",
    "bb20_pos",
    "donchian55_pos",
    "mfi_14",
    "rsi_14",
    "vwap_dist_pct",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
STATE_FEATURES = list(dict.fromkeys([*REGIME_FEATURES, *LOCAL_STATE_FEATURES]))
POLICIES = ("baseline_logistic", "confidence_trust_q65", "local_state_trust_q65")


def base_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
    ])


def gate_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                C=0.3,
                random_state=43,
            ),
        ),
    ])


def _base_confidence(model: Pipeline, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    return np.max(probabilities, axis=1)


def _inner_oof_meta(train: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    n = len(train)
    first_test = max(50000, n // 2)
    remaining = n - first_test
    fold_size = remaining // INNER_FOLDS
    if fold_size < 3000:
        raise RuntimeError(f"insufficient inner-fold span: n={n} fold_size={fold_size}")

    parts: list[pd.DataFrame] = []
    for fold in range(INNER_FOLDS):
        test_start = first_test + fold * fold_size
        test_end = n if fold == INNER_FOLDS - 1 else first_test + (fold + 1) * fold_size
        fit_end = test_start - HORIZON
        if fit_end < 50000 or test_end - test_start < 3000:
            raise RuntimeError(
                f"invalid inner fold {fold}: fit_end={fit_end} test_rows={test_end-test_start}"
            )
        fit = train.iloc[:fit_end]
        hold = train.iloc[test_start:test_end]
        y_fit = fit["target"].astype(int).to_numpy()
        if len(np.unique(y_fit)) < 3:
            raise RuntimeError(f"inner fold {fold} lacks three target classes")
        model = base_model().fit(fit[base_features].to_numpy(float), y_fit)
        x_hold = hold[base_features].to_numpy(float)
        pred = model.predict(x_hold).astype(int)
        confidence = _base_confidence(model, x_hold)
        part = hold[["timestamp", *STATE_FEATURES]].copy()
        part["base_pred"] = pred
        part["base_confidence"] = confidence
        part["correct"] = (pred == hold["target"].astype(int).to_numpy()).astype(int)
        part["inner_fold"] = fold
        parts.append(part)

    meta = pd.concat(parts, ignore_index=True)
    if len(meta) < MIN_META_ROWS:
        raise RuntimeError(f"insufficient chronology-clean meta rows: {len(meta)}")
    if meta["correct"].nunique() < 2:
        raise RuntimeError("meta correctness target is single-class")
    return meta


def _tail_summary(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < MIN_COMPLETE_WEEKS:
        raise RuntimeError(f"insufficient weekly economics: {len(arr)}")
    k = max(1, int(np.ceil(0.10 * len(arr))))
    worst = np.sort(arr)[:k]
    cumulative = np.cumsum(arr)
    curve = np.concatenate(([0.0], cumulative))
    peak = np.maximum.accumulate(curve)
    drawdown = curve - peak
    positive_total = float(np.sum(arr[arr > 0]))
    largest_positive = np.sort(arr[arr > 0])[-5:] if np.any(arr > 0) else np.asarray([], dtype=float)
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
        "top5_positive_share": (
            float(np.sum(largest_positive) / positive_total) if positive_total > 0 else None
        ),
    }


def summarize(rows: list[dict], policy: str) -> dict:
    result = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        values = [row["policies"][policy]["phase_audit"].get(field) for row in rows]
        result[f"after_{key}pt"] = _tail_summary(np.asarray(values, dtype=float))
    coverage = np.asarray([row["policies"][policy]["coverage"] for row in rows], dtype=float)
    result["coverage"] = {
        "median": float(np.median(coverage)),
        "mean": float(np.mean(coverage)),
        "min": float(np.min(coverage)),
        "max": float(np.max(coverage)),
    }
    return result


def paired(rows: list[dict], challenger: str) -> dict:
    result = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        diffs = []
        for row in rows:
            a = row["policies"][challenger]["phase_audit"].get(field)
            b = row["policies"]["baseline_logistic"]["phase_audit"].get(field)
            if a is not None and b is not None:
                diffs.append(float(a) - float(b))
        arr = np.asarray(diffs, dtype=float)
        if len(arr) < MIN_COMPLETE_WEEKS:
            raise RuntimeError(f"insufficient paired weeks for {challenger}/{key}: {len(arr)}")
        k = max(1, int(np.ceil(0.10 * len(arr))))
        worst = np.sort(arr)[:k]
        result[f"after_{key}pt"] = {
            "weeks": int(len(arr)),
            "median_delta_points": float(np.median(arr)),
            "mean_delta_points": float(np.mean(arr)),
            "p10_delta_points": float(np.quantile(arr, 0.10)),
            "bottom10pct_mean_delta_points": float(np.mean(worst)),
            "win_fraction": float(np.mean(arr > 0)),
            "loss_fraction": float(np.mean(arr < 0)),
            "tie_fraction": float(np.mean(arr == 0)),
        }
    return result


def _decision(summary: dict, paired_result: dict, challenger: str) -> dict:
    reasons = []
    for cost_key in ("after_1p0pt", "after_2p0pt"):
        base = summary["baseline_logistic"][cost_key]
        helper = summary[challenger][cost_key]
        delta = paired_result[challenger][cost_key]
        checks = {
            "paired_median_delta_positive": delta["median_delta_points"] > 0,
            "paired_win_fraction_above_half": delta["win_fraction"] > 0.50,
            "positive_week_fraction_not_worse": helper["positive_week_fraction"] >= base["positive_week_fraction"],
            "p10_not_worse": helper["p10_weekly_phase_median_points"] >= base["p10_weekly_phase_median_points"],
            "bottom10pct_mean_not_worse": helper["bottom10pct_mean_points"] >= base["bottom10pct_mean_points"],
            "max_drawdown_not_worse": helper["max_drawdown_weekly_phase_median_points"] >= base["max_drawdown_weekly_phase_median_points"],
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            reasons.append({"cost": cost_key, "failed": failed})
    return {
        "action": "advance_to_nq_external_validation" if not reasons else "reject_or_rework_helper",
        "rule": "must pass every predeclared paired weekly-economic and tail check at both 1pt and 2pt costs",
        "failures": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    base_features = list(BASE_FEATURES)
    all_features = list(dict.fromkeys([*base_features, *STATE_FEATURES]))
    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *all_features]))
    work = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, _, _ = target_columns(work, HORIZON, VOL_MULTIPLIER)
    work["target"] = label
    work["point_move"] = work["close"].shift(-HORIZON) - work["close"]
    work["trade_week"] = trade_week_key(work["timestamp"])

    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    weekly_rows: list[dict] = []
    fit_receipts: list[dict] = []

    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        mask = (
            (work["timestamp"] >= start)
            & (work["timestamp"] < end)
            & work["target"].notna()
            & work["point_move"].notna()
        )
        test_idx = np.flatnonzero(mask.to_numpy())
        if len(test_idx) < 2000:
            continue
        outer_test_start = int(test_idx[0])
        outer_train_end = outer_test_start - HORIZON
        if outer_train_end < 70000:
            continue
        train = work.iloc[:outer_train_end]
        train = train[train["target"].notna()].copy()
        test = work.iloc[int(test_idx[0]) : int(test_idx[-1] + 1)]
        test = test[
            (test["timestamp"] >= start)
            & (test["timestamp"] < end)
            & test["target"].notna()
            & test["point_move"].notna()
        ].copy()
        if len(train) < 70000 or len(test) < 2000:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("outer quarter chronology overlap")

        y_train = train["target"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 3:
            continue

        meta = _inner_oof_meta(train, base_features)
        gate_features = [*STATE_FEATURES, "base_confidence", "base_pred"]
        gate = gate_model().fit(meta[gate_features].to_numpy(float), meta["correct"].to_numpy(int))
        meta_gate_prob = gate.predict_proba(meta[gate_features].to_numpy(float))[:, 1]
        gate_cut = float(np.quantile(meta_gate_prob, TRUST_QUANTILE))
        confidence_cut = float(np.quantile(meta["base_confidence"].to_numpy(float), TRUST_QUANTILE))

        base = base_model().fit(train[base_features].to_numpy(float), y_train)
        fit_receipts.append(
            {
                "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
                "train_rows": int(len(train)),
                "meta_rows": int(len(meta)),
                "meta_inner_folds": int(meta["inner_fold"].nunique()),
                "train_last_timestamp": train["timestamp"].max().isoformat(),
                "test_rows": int(len(test)),
                "test_first_timestamp": test["timestamp"].min().isoformat(),
                "test_last_timestamp": test["timestamp"].max().isoformat(),
                "confidence_q65": confidence_cut,
                "local_state_gate_q65": gate_cut,
            }
        )

        for week_key, positions in test.groupby("trade_week", sort=True).groups.items():
            pos = np.asarray(list(positions), dtype=int)
            week = work.loc[pos]
            week = week[
                (week["timestamp"] >= start)
                & (week["timestamp"] < end)
                & week["target"].notna()
                & week["point_move"].notna()
            ].copy()
            if len(week) < 300:
                continue

            x_week = week[base_features].to_numpy(float)
            baseline_pred = base.predict(x_week).astype(int)
            baseline_conf = _base_confidence(base, x_week)
            gate_frame = week[STATE_FEATURES].copy()
            gate_frame["base_confidence"] = baseline_conf
            gate_frame["base_pred"] = baseline_pred
            gate_prob = gate.predict_proba(gate_frame[gate_features].to_numpy(float))[:, 1]

            confidence_pred = np.where(baseline_conf >= confidence_cut, baseline_pred, 0).astype(int)
            state_pred = np.where(gate_prob >= gate_cut, baseline_pred, 0).astype(int)
            predictions = {
                "baseline_logistic": baseline_pred,
                "confidence_trust_q65": confidence_pred,
                "local_state_trust_q65": state_pred,
            }

            y_week = week["target"].astype(int).to_numpy()
            point_move = week["point_move"].to_numpy(float)
            policies = {}
            for name, pred in predictions.items():
                policies[name] = {
                    "classification": classification(y_week, pred),
                    "coverage": float(np.mean(pred != 0)),
                    "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "phase_audit": phase_audit(week["timestamp"], pred, point_move, HORIZON),
                }
            weekly_rows.append(
                {
                    "trade_week": pd.Timestamp(week_key).isoformat(),
                    "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
                    "rows": int(len(week)),
                    "policies": policies,
                }
            )

    by_week: dict[str, dict] = {}
    for row in weekly_rows:
        key = row["trade_week"]
        if key in by_week:
            if row["rows"] == by_week[key]["rows"]:
                raise RuntimeError(f"ambiguous duplicate trade week {key}")
            if row["rows"] > by_week[key]["rows"]:
                by_week[key] = row
        else:
            by_week[key] = row
    rows = [by_week[key] for key in sorted(by_week)]
    if len(rows) < MIN_COMPLETE_WEEKS:
        raise RuntimeError(f"insufficient unique weekly OOS rows: {len(rows)}")

    summaries = {name: summarize(rows, name) for name in POLICIES}
    paired_results = {
        name: paired(rows, name) for name in POLICIES if name != "baseline_logistic"
    }
    decisions = {
        name: _decision(summaries, paired_results, name)
        for name in POLICIES
        if name != "baseline_logistic"
    }

    result = {
        "schema": "foundry.mnq_h12_helper_weekly_economics.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "timeframe": "12Min",
        "horizon": HORIZON,
        "vol_multiplier": VOL_MULTIPLIER,
        "minimum_complete_weeks": MIN_COMPLETE_WEEKS,
        "policies": list(POLICIES),
        "base_feature_contract": "baseline20",
        "helper_feature_contract": {
            "state_features": STATE_FEATURES,
            "gate_inputs": [*STATE_FEATURES, "base_confidence", "base_pred"],
            "gate_model": "StandardScaler + class-balanced LogisticRegression(C=.3, random_state=43)",
            "gate_action": "abstain only; never invert or replace base direction",
            "trust_quantile": TRUST_QUANTILE,
        },
        "protocol": (
            "fixed quarterly outer walk-forward; horizon purge; base logistic fit only on prior MNQ; "
            "helper correctness model trained only on chronology-clean inner-OOF base predictions from prior data; "
            "q65 cuts learned only from inner-OOF prior rows; identical OOS MNQ weeks and phase audit for every policy; "
            "all non-overlapping UTC phase streams; costs 0.5/1/2/4 MNQ points; no OOS threshold tuning, no direction inversion, "
            "no NQ contemporaneous helper inputs, no post-hoc week or regime selection"
        ),
        "fit_receipts": fit_receipts,
        "weekly_rows": rows,
        "summary": summaries,
        "paired_vs_baseline": paired_results,
        "predeclared_decision": decisions,
        "nq_validation_boundary": (
            "NQ is reserved for external/pre-2019 relationship validation only after an MNQ helper passes the frozen MNQ rule; "
            "NQ does not train or tune this MNQ helper"
        ),
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H12_HELPER_WEEKLY_ECONOMICS=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print(json.dumps(result["paired_vs_baseline"], sort_keys=True))
    print(json.dumps(result["predeclared_decision"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
