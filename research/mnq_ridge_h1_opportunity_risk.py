from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research.mnq_ridge_h1_risk_budget as base
from research.expanded_regime_ablation import BASE_FEATURES
from research.mnq_ridge_h1_risk_budget_warmupfixed import join_risk_scores_available

CONSUMER = "opportunity_to_mae_exposure_neutral_budget"
CONTRACT = "research/mnq_ridge_h1_opportunity_risk_contract_20260901.json"


def opportunity_risk_raw_weight(
    signal: np.ndarray,
    abs_ridge_pred_z: np.ndarray,
    long_pred_mae: np.ndarray,
    short_pred_mae: np.ndarray,
    references: dict[str, dict[str, float]],
) -> np.ndarray:
    out = np.zeros(len(signal), dtype=float)
    for side, label, risk_values in (
        (1, "long", long_pred_mae),
        (-1, "short", short_pred_mae),
    ):
        mask = signal == side
        if not np.any(mask):
            continue
        opp_ref = float(references[label]["median_abs_ridge_pred_z"])
        risk_ref = float(references[label]["median_predicted_mae_z"])
        if opp_ref <= 0 or risk_ref <= 0:
            raise RuntimeError(f"invalid {label} opportunity/risk reference")
        opportunity_intensity = np.maximum(abs_ridge_pred_z[mask], 1e-12) / opp_ref
        risk_intensity = np.maximum(risk_values[mask], 1e-12) / risk_ref
        quality = opportunity_intensity / np.maximum(risk_intensity, 1e-12)
        out[mask] = np.clip(quality, base.WEIGHT_MIN, base.WEIGHT_MAX)
    return out


def fit_quarter(
    op: pd.DataFrame,
    risk: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict] | None:
    features = list(BASE_FEATURES)
    op_mask = (
        (op["timestamp"] >= start)
        & (op["timestamp"] < end)
        & op["target_move_z"].notna()
        & op["point_move"].notna()
    )
    op_idx = np.flatnonzero(op_mask.to_numpy())
    if len(op_idx) < 2000:
        return None
    first_test = int(op_idx[0])
    train_end = first_test - base.RIDGE_HORIZON
    if train_end < 50_000:
        return None

    op_train = op.iloc[:train_end].copy()
    op_train = op_train[op_train["target_move_z"].notna()].copy()
    op_test = op.iloc[int(op_idx[0]) : int(op_idx[-1] + 1)].copy()
    op_test = op_test[
        (op_test["timestamp"] >= start)
        & (op_test["timestamp"] < end)
        & op_test["target_move_z"].notna()
        & op_test["point_move"].notna()
    ].copy()
    if op_train["timestamp"].max() >= op_test["timestamp"].min():
        raise RuntimeError("Ridge opportunity-risk chronology violation")

    x_train = op_train[features].to_numpy(float)
    y_train = op_train["target_move_z"].to_numpy(float)
    ridge_threshold, ridge_oof = base.chronological_oof_threshold(
        "ridge", x_train, y_train, base.RIDGE_HORIZON
    )
    ridge = base.make_regressor("ridge").fit(x_train, y_train)
    train_pred_z = np.asarray(ridge.predict(x_train), dtype=float)
    test_pred_z = np.asarray(ridge.predict(op_test[features].to_numpy(float)), dtype=float)
    train_signal = np.where(
        np.abs(train_pred_z) >= ridge_threshold, np.sign(train_pred_z), 0
    ).astype(int)
    test_signal = np.where(
        np.abs(test_pred_z) >= ridge_threshold, np.sign(test_pred_z), 0
    ).astype(int)

    risk_train = risk[
        risk["timestamp"] < start - pd.Timedelta(hours=base.RISK_HORIZON)
    ].dropna(subset=list(base.RISK_TARGETS)).copy()
    if len(risk_train) < 5000:
        return None
    rx = risk_train[base.RISK_FEATURES].to_numpy(float)
    risk_scores = risk.copy()
    sx = risk_scores[base.RISK_FEATURES].to_numpy(float)
    for target in ("long_mae_z", "short_mae_z"):
        model = base.risk_model().fit(rx, risk_train[target].to_numpy(float))
        risk_scores[f"pred_{target}"] = model.predict(sx)

    train_for_join = op_train.copy()
    train_for_join["_ridge_signal"] = train_signal
    train_for_join["_ridge_abs_pred_z"] = np.abs(train_pred_z)
    train_join = join_risk_scores_available(train_for_join, risk_scores, strict=False)
    available_train_signal = train_join.pop("_ridge_signal").to_numpy(int)
    available_train_abs_pred = train_join.pop("_ridge_abs_pred_z").to_numpy(float)

    test_for_join = op_test.copy()
    test_for_join["_ridge_signal"] = test_signal
    test_for_join["_ridge_abs_pred_z"] = np.abs(test_pred_z)
    test_join = join_risk_scores_available(test_for_join, risk_scores, strict=True)
    available_test_signal = test_join.pop("_ridge_signal").to_numpy(int)
    available_test_abs_pred = test_join.pop("_ridge_abs_pred_z").to_numpy(float)
    if len(test_join) != len(op_test):
        raise RuntimeError("strict OOS risk availability changed Ridge decision support")

    references: dict[str, dict[str, float]] = {}
    for side, label, target in (
        (1, "long", "long_mae_z"),
        (-1, "short", "short_mae_z"),
    ):
        mask = available_train_signal == side
        risk_vals = train_join.loc[mask, f"pred_{target}"].to_numpy(float)
        opp_vals = available_train_abs_pred[mask]
        if len(risk_vals) < 1000:
            raise RuntimeError(
                f"insufficient prior {label} Ridge signals for opportunity-risk reference: {len(risk_vals)}"
            )
        references[label] = {
            "signals": int(len(risk_vals)),
            "median_abs_ridge_pred_z": float(np.median(opp_vals)),
            "median_predicted_mae_z": float(np.median(risk_vals)),
        }

    train_raw = opportunity_risk_raw_weight(
        available_train_signal,
        available_train_abs_pred,
        train_join["pred_long_mae_z"].to_numpy(float),
        train_join["pred_short_mae_z"].to_numpy(float),
        references,
    )
    train_selected = available_train_signal != 0
    normalization = float(np.mean(train_raw[train_selected]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise RuntimeError("invalid training-only opportunity-risk normalization")

    test_raw = opportunity_risk_raw_weight(
        available_test_signal,
        available_test_abs_pred,
        test_join["pred_long_mae_z"].to_numpy(float),
        test_join["pred_short_mae_z"].to_numpy(float),
        references,
    )
    weights = np.zeros(len(available_test_signal), dtype=float)
    selected = available_test_signal != 0
    weights[selected] = np.clip(
        test_raw[selected] / normalization, base.WEIGHT_MIN, base.WEIGHT_MAX
    )

    test_join["baseline_signal"] = available_test_signal
    test_join["risk_budget_weight"] = weights
    quarter = f"{start.year}Q{((start.month - 1) // 3) + 1}"
    test_join["quarter"] = quarter

    selected_weights = weights[selected]
    long_weights = weights[available_test_signal == 1]
    short_weights = weights[available_test_signal == -1]
    receipt = {
        "quarter": quarter,
        "consumer": CONSUMER,
        "ridge_train_rows": int(len(op_train)),
        "risk_available_ridge_train_rows": int(len(train_join)),
        "risk_warmup_ridge_train_rows_excluded": int(len(op_train) - len(train_join)),
        "ridge_test_rows": int(len(op_test)),
        "strict_oos_risk_rows": int(len(test_join)),
        "ridge_train_last_timestamp": op_train["timestamp"].max().isoformat(),
        "ridge_oof": ridge_oof,
        "risk_train_rows": int(len(risk_train)),
        "risk_train_last_timestamp": risk_train["timestamp"].max().isoformat(),
        "side_training_references": references,
        "training_clipped_quality_mean_normalization": normalization,
        "signals": int(np.sum(selected)),
        "long_signals": int(np.sum(available_test_signal == 1)),
        "short_signals": int(np.sum(available_test_signal == -1)),
        "weight": {
            "mean": float(np.mean(selected_weights)) if len(selected_weights) else None,
            "median": float(np.median(selected_weights)) if len(selected_weights) else None,
            "p10": float(np.quantile(selected_weights, 0.10)) if len(selected_weights) else None,
            "p90": float(np.quantile(selected_weights, 0.90)) if len(selected_weights) else None,
            "min": float(np.min(selected_weights)) if len(selected_weights) else None,
            "max": float(np.max(selected_weights)) if len(selected_weights) else None,
            "sum": float(np.sum(selected_weights)),
        },
        "long_weight_mean": float(np.mean(long_weights)) if len(long_weights) else None,
        "short_weight_mean": float(np.mean(short_weights)) if len(short_weights) else None,
        "risk_age_minutes": {
            "median": float(test_join["risk_age_minutes"].median()),
            "p95": float(test_join["risk_age_minutes"].quantile(0.95)),
            "max": float(test_join["risk_age_minutes"].max()),
        },
    }
    return test_join, receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--risk-prereq-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prereq = base.verify_prerequisite(args.risk_prereq_receipt)
    raw = base.load_deep(args.deep_root)
    stitched = base.stitch_deep(raw, base.deep_roll_schedule(raw))
    op = base.opportunity_matrix(stitched)
    risk = base.build_risk_matrix(base.bars_1h(stitched))

    quarter_starts = list(
        pd.date_range(base.DISCOVERY_START, base.DISCOVERY_END, freq="QS", tz="UTC")
    )
    rows: list[dict] = []
    fits: list[dict] = []
    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        fitted = fit_quarter(op, risk, start, end)
        if fitted is None:
            continue
        joined, receipt = fitted
        rows.extend(base.weekly_rows(joined))
        fits.append(receipt)
        print("QUARTER", receipt["quarter"], json.dumps(receipt, sort_keys=True))

    by_week: dict[str, dict] = {}
    for row in rows:
        key = row["trade_week"]
        if key not in by_week or row["rows"] > by_week[key]["rows"]:
            by_week[key] = row
    rows = [by_week[key] for key in sorted(by_week)]

    paired = {str(cost): base.paired_summary(rows, cost) for cost in base.POINT_COSTS}
    exposure = base.exposure_summary(fits)
    checks = {
        "1.0": base.advance_checks(paired["1.0"], exposure),
        "2.0": base.advance_checks(paired["2.0"], exposure),
    }
    passed = checks["1.0"]["passed"] and checks["2.0"]["passed"]

    result = {
        "schema": "foundry.mnq_ridge_h1_opportunity_risk_discovery.v1",
        "research_only": True,
        "promotion_authority": False,
        "contract": CONTRACT,
        "source": f"mbytes21/MNQ_DATA@{base.SOURCE_COMMIT}",
        "deep_timestamp_contract": base.contract_receipt(),
        "risk_prerequisite_commit": base.RISK_PREREQ_COMMIT,
        "risk_prerequisite": prereq,
        "consumer": CONSUMER,
        "weight_bounds": [base.WEIGHT_MIN, base.WEIGHT_MAX],
        "discovery_period": {
            "start": base.DISCOVERY_START.isoformat(),
            "end_exclusive": base.DISCOVERY_END.isoformat(),
        },
        "fit_receipts": fits,
        "weekly_rows": rows,
        "paired_summary": paired,
        "exposure": exposure,
        "advance_checks": checks,
        "decision": (
            "advance_unchanged_to_corrected_2026_confirmation"
            if passed
            else "reject_as_specified"
        ),
        "decomposition_after_1pt": {
            "year": base.decomposition(rows, 1.0, "year"),
            "quarter": base.decomposition(rows, 1.0, "quarter"),
        },
        "decomposition_after_2pt": {
            "year": base.decomposition(rows, 2.0, "year"),
            "quarter": base.decomposition(rows, 2.0, "quarter"),
        },
        "nq_boundary": "NQ remains excluded until unchanged MNQ discovery and corrected-2026 confirmation both pass.",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_RIDGE_H1_OPPORTUNITY_RISK_DISCOVERY=PASS")
    print("DECISION=" + result["decision"])
    print("EXPOSURE=" + json.dumps(exposure, sort_keys=True))
    print("CHECKS=" + json.dumps(checks, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
