from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.expanded_regime_ablation import BASE_FEATURES
from research.mnq_expected_move_regression import chronological_oof_threshold, make_regressor
from research.mnq_external_transfer_validation import deep_roll_schedule, stitch_deep
from research.mnq_ridge_h1_mae_orchestration import (
    DISCOVERY_END,
    DISCOVERY_START,
    RIDGE_HORIZON,
    opportunity_matrix,
)
from research.nq_to_mnq_execution_transfer import POINT_COSTS, phase_audit

CONTRACT = "research/mnq_h12_ridge_authority_audit_contract_20260901.json"
POLICIES = (
    "ridge",
    "same_selected_always_long",
    "same_selected_prior_majority",
    "ridge_long_only",
    "ridge_short_only",
    "sign_flipped_ridge",
)
MIN_WEEK_ROWS = 300


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float | None:
    x = pd.Series(np.asarray(a, dtype=float))
    y = pd.Series(np.asarray(b, dtype=float))
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 100 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return None
    return float(x[mask].corr(y[mask], method=method))


def _selection_diagnostics(pred: np.ndarray, truth_z: np.ndarray, point_move: np.ndarray, threshold: float) -> dict:
    pred = np.asarray(pred, dtype=float)
    truth_z = np.asarray(truth_z, dtype=float)
    point_move = np.asarray(point_move, dtype=float)
    selected = np.abs(pred) >= threshold
    nonzero_truth = truth_z != 0
    sign_eval = selected & nonzero_truth
    signed_z = np.sign(pred[selected]) * truth_z[selected]
    signed_points = np.sign(pred[selected]) * point_move[selected]
    same_long_points = point_move[selected]
    return {
        "rows": int(len(pred)),
        "selected_rows": int(np.sum(selected)),
        "coverage": float(np.mean(selected)),
        "long_selected": int(np.sum(selected & (pred > 0))),
        "short_selected": int(np.sum(selected & (pred < 0))),
        "raw_pearson": _corr(pred, truth_z, "pearson"),
        "raw_spearman": _corr(pred, truth_z, "spearman"),
        "abs_prediction_abs_truth_spearman": _corr(np.abs(pred), np.abs(truth_z), "spearman"),
        "selected_sign_accuracy": (
            float(np.mean(np.sign(pred[sign_eval]) == np.sign(truth_z[sign_eval]))) if np.any(sign_eval) else None
        ),
        "selected_signed_normalized_mean": float(np.mean(signed_z)) if len(signed_z) else None,
        "selected_signed_normalized_median": float(np.median(signed_z)) if len(signed_z) else None,
        "selected_signed_points_mean": float(np.mean(signed_points)) if len(signed_points) else None,
        "selected_signed_points_median": float(np.median(signed_points)) if len(signed_points) else None,
        "same_selected_always_long_points_mean": float(np.mean(same_long_points)) if len(same_long_points) else None,
        "same_selected_always_long_points_median": float(np.median(same_long_points)) if len(same_long_points) else None,
    }


def _oof_diagnostics(x: np.ndarray, y: np.ndarray, horizon: int) -> tuple[float, dict]:
    threshold, authority = chronological_oof_threshold("ridge", x, y, horizon)
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    receipts: list[dict] = []
    for i in range(4):
        test_start = first + i * fold
        test_end = n if i == 3 else first + (i + 1) * fold
        train_end = test_start - horizon
        model = make_regressor("ridge").fit(x[:train_end], y[:train_end])
        pred = np.asarray(model.predict(x[test_start:test_end]), dtype=float)
        target = np.asarray(y[test_start:test_end], dtype=float)
        preds.append(pred)
        truth.append(target)
        receipts.append({
            "fold": i,
            "train_rows": int(train_end),
            "test_rows": int(test_end - test_start),
            "prediction_target_pearson": _corr(pred, target, "pearson"),
            "prediction_target_spearman": _corr(pred, target, "spearman"),
        })
    p = np.concatenate(preds)
    t = np.concatenate(truth)
    reconstructed = float(np.quantile(np.abs(p), 0.50))
    if not np.isclose(threshold, reconstructed, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"OOF threshold reproduction mismatch: authority={threshold} reconstructed={reconstructed}")
    selected = np.abs(p) >= threshold
    nonzero_truth = t != 0
    sign_eval = selected & nonzero_truth
    signed = np.sign(p[selected]) * t[selected]
    out = {
        **authority,
        "reproduced_threshold_abs_pred_z": reconstructed,
        "raw_spearman": _corr(p, t, "spearman"),
        "abs_prediction_abs_truth_spearman": _corr(np.abs(p), np.abs(t), "spearman"),
        "selected_rows": int(np.sum(selected)),
        "selected_coverage": float(np.mean(selected)),
        "selected_long": int(np.sum(selected & (p > 0))),
        "selected_short": int(np.sum(selected & (p < 0))),
        "selected_sign_accuracy": (
            float(np.mean(np.sign(p[sign_eval]) == np.sign(t[sign_eval]))) if np.any(sign_eval) else None
        ),
        "selected_signed_normalized_mean": float(np.mean(signed)) if len(signed) else None,
        "selected_signed_normalized_median": float(np.median(signed)) if len(signed) else None,
        "fold_diagnostics": receipts,
    }
    return threshold, out


def _signals(ridge: np.ndarray, prior_majority_sign: int) -> dict[str, np.ndarray]:
    selected = ridge != 0
    return {
        "ridge": ridge.copy(),
        "same_selected_always_long": np.where(selected, 1, 0).astype(int),
        "same_selected_prior_majority": np.where(selected, prior_majority_sign, 0).astype(int),
        "ridge_long_only": np.where(ridge == 1, 1, 0).astype(int),
        "ridge_short_only": np.where(ridge == -1, -1, 0).astype(int),
        "sign_flipped_ridge": (-ridge).astype(int),
    }


def _weekly_rows(frame: pd.DataFrame, policy_signals: dict[str, np.ndarray], quarter: str) -> list[dict]:
    work = frame.copy()
    work["_pos"] = np.arange(len(work))
    rows: list[dict] = []
    for week_key, group in work.groupby("trade_week", sort=True):
        if len(group) < MIN_WEEK_ROWS:
            continue
        pos = group["_pos"].to_numpy(int)
        move = group["point_move"].to_numpy(float)
        policies = {}
        for name in POLICIES:
            sig = policy_signals[name][pos]
            policies[name] = {
                "coverage": float(np.mean(sig != 0)),
                "long_signals": int(np.sum(sig == 1)),
                "short_signals": int(np.sum(sig == -1)),
                "phase_audit": phase_audit(group["timestamp"], sig, move, RIDGE_HORIZON),
            }
        rows.append({
            "trade_week": pd.Timestamp(week_key).isoformat(),
            "quarter": quarter,
            "rows": int(len(group)),
            "policies": policies,
        })
    return rows


def _tail_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
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
        "worst": float(np.min(arr)),
        "best": float(np.max(arr)),
        "cumulative_points": float(np.sum(arr)),
        "max_drawdown_points": float(np.min(draw)),
    }


def _weekly_values(rows: list[dict], policy: str, cost: float) -> tuple[np.ndarray, list[str]]:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    values: list[float] = []
    labels: list[str] = []
    for row in rows:
        value = row["policies"][policy]["phase_audit"].get(field)
        if value is not None and np.isfinite(value):
            values.append(float(value))
            labels.append(row["trade_week"])
    return np.asarray(values, dtype=float), labels


def _policy_summary(rows: list[dict], policy: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        values, labels = _weekly_values(rows, policy, cost)
        key = str(cost)
        out[key] = _tail_stats(values) if len(values) else {"weeks": 0}
        if labels:
            out[key]["first_week"] = labels[0]
            out[key]["last_week"] = labels[-1]
    return out


def _paired(rows: list[dict], challenger: str, control: str, cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    a: list[float] = []
    b: list[float] = []
    labels: list[str] = []
    for row in rows:
        av = row["policies"][challenger]["phase_audit"].get(field)
        bv = row["policies"][control]["phase_audit"].get(field)
        if av is None or bv is None or not np.isfinite(av) or not np.isfinite(bv):
            continue
        a.append(float(av)); b.append(float(bv)); labels.append(row["trade_week"])
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    delta = aa - bb
    if not len(delta):
        return {"weeks": 0}
    return {
        "weeks": int(len(delta)),
        "ridge": _tail_stats(aa),
        "control": _tail_stats(bb),
        "delta": {
            "median": float(np.median(delta)),
            "mean": float(np.mean(delta)),
            "win_fraction": float(np.mean(delta > 0)),
            "tie_fraction": float(np.mean(delta == 0)),
            "p10": float(np.quantile(delta, 0.10)),
            "worst": float(np.min(delta)),
            "best": float(np.max(delta)),
        },
        "first_week": labels[0],
        "last_week": labels[-1],
    }


def _directional_gate(paired: dict[str, dict[str, dict]]) -> dict:
    checks: dict[str, bool] = {}
    for control in ("same_selected_always_long", "same_selected_prior_majority"):
        for cost in (1.0, 2.0):
            rec = paired[control][str(cost)]
            prefix = f"{control}_after_{str(cost).replace('.', 'p')}pt"
            checks[f"{prefix}_paired_mean_positive"] = rec["delta"]["mean"] > 0
            checks[f"{prefix}_paired_median_positive"] = rec["delta"]["median"] > 0
            checks[f"{prefix}_paired_win_fraction_gt_0p50"] = rec["delta"]["win_fraction"] > 0.50
            checks[f"{prefix}_ridge_mean_gt_control"] = rec["ridge"]["mean"] > rec["control"]["mean"]
            checks[f"{prefix}_ridge_median_gt_control"] = rec["ridge"]["median"] > rec["control"]["median"]
    passed = all(checks.values())
    return {
        "passed": passed,
        "interpretation": (
            "directional_opportunity_interpretation_survives_audit"
            if passed
            else "do_not_describe_current_h12_ridge_as_established_directional_alpha"
        ),
        "checks": checks,
        "failed": [name for name, ok in checks.items() if not ok],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    op = opportunity_matrix(stitched)
    features = list(BASE_FEATURES)

    quarter_starts = list(pd.date_range(DISCOVERY_START, DISCOVERY_END, freq="QS", tz="UTC"))
    rows: list[dict] = []
    fits: list[dict] = []

    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        mask = (
            (op["timestamp"] >= start)
            & (op["timestamp"] < end)
            & op["target_move_z"].notna()
            & op["point_move"].notna()
        )
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 2000:
            continue
        test_start = int(idx[0])
        train_end = test_start - RIDGE_HORIZON
        if train_end < 50_000:
            continue
        train = op.iloc[:train_end].copy()
        train = train[train["target_move_z"].notna()].copy()
        test = op.iloc[int(idx[0]) : int(idx[-1] + 1)].copy()
        test = test[
            (test["timestamp"] >= start)
            & (test["timestamp"] < end)
            & test["target_move_z"].notna()
            & test["point_move"].notna()
        ].copy()
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("H12 Ridge authority audit chronology violation")

        x_train = train[features].to_numpy(float)
        y_train = train["target_move_z"].to_numpy(float)
        threshold, oof = _oof_diagnostics(x_train, y_train, RIDGE_HORIZON)
        ridge_model = make_regressor("ridge").fit(x_train, y_train)
        test_pred = np.asarray(ridge_model.predict(test[features].to_numpy(float)), dtype=float)
        ridge = np.where(np.abs(test_pred) >= threshold, np.sign(test_pred), 0).astype(int)
        prior_majority_sign = 1 if float(np.mean(y_train)) >= 0 else -1
        policies = _signals(ridge, prior_majority_sign)
        quarter = f"{start.year}Q{((start.month - 1) // 3) + 1}"

        rows.extend(_weekly_rows(test, policies, quarter))
        fits.append({
            "quarter": quarter,
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "threshold_abs_pred_z": threshold,
            "prior_training_target_mean": float(np.mean(y_train)),
            "prior_majority_sign": prior_majority_sign,
            "oof": oof,
            "outer": _selection_diagnostics(
                test_pred,
                test["target_move_z"].to_numpy(float),
                test["point_move"].to_numpy(float),
                threshold,
            ),
        })
        print("QUARTER", quarter, json.dumps(fits[-1], sort_keys=True))

    by_week: dict[str, dict] = {}
    for row in rows:
        key = row["trade_week"]
        if key not in by_week or row["rows"] > by_week[key]["rows"]:
            by_week[key] = row
    rows = [by_week[key] for key in sorted(by_week)]
    if len(rows) < 80:
        raise RuntimeError(f"insufficient H12 Ridge audit weeks: {len(rows)}")

    policy_summary = {policy: _policy_summary(rows, policy) for policy in POLICIES}
    paired = {
        control: {str(cost): _paired(rows, "ridge", control, cost) for cost in POINT_COSTS}
        for control in ("same_selected_always_long", "same_selected_prior_majority")
    }
    gate = _directional_gate(paired)

    result = {
        "schema": "foundry.mnq_h12_ridge_authority_audit.v1",
        "research_only": True,
        "promotion_authority": False,
        "contract": CONTRACT,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "deep_timestamp_contract": contract_receipt(),
        "policy": "unchanged 12Min/H12 baseline20 Ridge(alpha=10) with prior-only inner-OOF q50 absolute-prediction gate",
        "fit_receipts": fits,
        "weekly_rows": rows,
        "policy_summary": policy_summary,
        "paired_vs_same_selection_controls": paired,
        "directional_increment_gate": gate,
        "side_decomposition": {
            "ridge_long_only": policy_summary["ridge_long_only"],
            "ridge_short_only": policy_summary["ridge_short_only"],
            "sign_flipped_ridge": policy_summary["sign_flipped_ridge"],
        },
        "next_authority": "audit only; no model or trading promotion",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H12_RIDGE_AUTHORITY_AUDIT=PASS")
    print("DIRECTIONAL_INCREMENT_GATE=" + json.dumps(gate, sort_keys=True))
    print("RIDGE_1PT=" + json.dumps(policy_summary["ridge"]["1.0"], sort_keys=True))
    print("ALWAYS_LONG_1PT=" + json.dumps(policy_summary["same_selected_always_long"]["1.0"], sort_keys=True))
    print("PRIOR_MAJORITY_1PT=" + json.dumps(policy_summary["same_selected_prior_majority"]["1.0"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
