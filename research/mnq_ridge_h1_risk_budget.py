from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.mnq_expected_move_regression import chronological_oof_threshold, make_regressor
from research.mnq_external_transfer_validation import deep_roll_schedule, stitch_deep
from research.mnq_ridge_h1_mae_orchestration import (
    DISCOVERY_END,
    DISCOVERY_START,
    RIDGE_HORIZON,
    RISK_FEATURES,
    RISK_HORIZON,
    RISK_TARGETS,
    bars_1h,
    build_risk_matrix,
    opportunity_matrix,
    risk_model,
    verify_prerequisite,
)
from research.nq_to_mnq_execution_transfer import POINT_COSTS

SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
RISK_PREREQ_COMMIT = "d2e1e0c4fd80235a76b0969a249f90eeb7b227d1"
WEIGHT_MIN = 0.50
WEIGHT_MAX = 1.50
MIN_PAIRED_WEEKS = 80


def join_risk_scores(op_frame: pd.DataFrame, risk_scores: pd.DataFrame) -> pd.DataFrame:
    joined = pd.merge_asof(
        op_frame.sort_values("timestamp"),
        risk_scores[["available_at", "pred_long_mae_z", "pred_short_mae_z"]].sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    needed = ["pred_long_mae_z", "pred_short_mae_z"]
    if joined[needed].isna().any().any():
        raise RuntimeError("missing completed-hour 1H risk score for Ridge decision")
    if (joined["available_at"] > joined["timestamp"]).any():
        raise RuntimeError("1H risk lookahead detected")
    joined["risk_age_minutes"] = (joined["timestamp"] - joined["available_at"]) / pd.Timedelta(minutes=1)
    return joined


def side_raw_weight(signal: np.ndarray, long_pred: np.ndarray, short_pred: np.ndarray, medians: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(signal), dtype=float)
    long_mask = signal == 1
    short_mask = signal == -1
    if np.any(long_mask):
        values = medians["long_mae_z"] / np.maximum(long_pred[long_mask], 1e-12)
        out[long_mask] = np.clip(values, WEIGHT_MIN, WEIGHT_MAX)
    if np.any(short_mask):
        values = medians["short_mae_z"] / np.maximum(short_pred[short_mask], 1e-12)
        out[short_mask] = np.clip(values, WEIGHT_MIN, WEIGHT_MAX)
    return out


def fit_quarter(op: pd.DataFrame, risk: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict] | None:
    features = [c for c in op.columns if c not in {"timestamp", "close", "target_move_z", "point_move", "trade_week"}]
    # Keep exactly the baseline20 feature authority used by the inherited opportunity_matrix.
    from research.expanded_regime_ablation import BASE_FEATURES
    features = list(BASE_FEATURES)

    op_mask = (op["timestamp"] >= start) & (op["timestamp"] < end) & op["target_move_z"].notna() & op["point_move"].notna()
    op_idx = np.flatnonzero(op_mask.to_numpy())
    if len(op_idx) < 2000:
        return None
    first_test = int(op_idx[0])
    train_end = first_test - RIDGE_HORIZON
    if train_end < 50_000:
        return None
    op_train = op.iloc[:train_end].copy()
    op_train = op_train[op_train["target_move_z"].notna()].copy()
    op_test = op.iloc[int(op_idx[0]): int(op_idx[-1] + 1)].copy()
    op_test = op_test[(op_test["timestamp"] >= start) & (op_test["timestamp"] < end) & op_test["target_move_z"].notna() & op_test["point_move"].notna()].copy()
    if op_train["timestamp"].max() >= op_test["timestamp"].min():
        raise RuntimeError("Ridge risk-budget chronology violation")

    x_train = op_train[features].to_numpy(float)
    y_train = op_train["target_move_z"].to_numpy(float)
    ridge_threshold, ridge_oof = chronological_oof_threshold("ridge", x_train, y_train, RIDGE_HORIZON)
    ridge = make_regressor("ridge").fit(x_train, y_train)
    train_pred_z = ridge.predict(x_train)
    test_pred_z = ridge.predict(op_test[features].to_numpy(float))
    train_signal = np.where(np.abs(train_pred_z) >= ridge_threshold, np.sign(train_pred_z), 0).astype(int)
    test_signal = np.where(np.abs(test_pred_z) >= ridge_threshold, np.sign(test_pred_z), 0).astype(int)

    risk_train = risk[(risk["timestamp"] < start - pd.Timedelta(hours=RISK_HORIZON))].dropna(subset=list(RISK_TARGETS)).copy()
    if len(risk_train) < 5000:
        return None
    rx = risk_train[RISK_FEATURES].to_numpy(float)
    risk_scores = risk.copy()
    sx = risk_scores[RISK_FEATURES].to_numpy(float)
    for target in ("long_mae_z", "short_mae_z"):
        model = risk_model().fit(rx, risk_train[target].to_numpy(float))
        risk_scores[f"pred_{target}"] = model.predict(sx)

    train_join = join_risk_scores(op_train, risk_scores)
    test_join = join_risk_scores(op_test, risk_scores)

    medians = {}
    for side, target in ((1, "long_mae_z"), (-1, "short_mae_z")):
        pred_col = f"pred_{target}"
        vals = train_join.loc[train_signal == side, pred_col].to_numpy(float)
        if len(vals) < 1000:
            raise RuntimeError(f"insufficient prior Ridge signals for {target} risk-budget reference: {len(vals)}")
        medians[target] = float(np.median(vals))

    train_raw = side_raw_weight(
        train_signal,
        train_join["pred_long_mae_z"].to_numpy(float),
        train_join["pred_short_mae_z"].to_numpy(float),
        medians,
    )
    train_selected = train_signal != 0
    normalization = float(np.mean(train_raw[train_selected]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise RuntimeError("invalid training-only risk-budget normalization")

    test_raw = side_raw_weight(
        test_signal,
        test_join["pred_long_mae_z"].to_numpy(float),
        test_join["pred_short_mae_z"].to_numpy(float),
        medians,
    )
    weights = np.zeros(len(test_signal), dtype=float)
    selected = test_signal != 0
    weights[selected] = np.clip(test_raw[selected] / normalization, WEIGHT_MIN, WEIGHT_MAX)

    test_join["baseline_signal"] = test_signal
    test_join["risk_budget_weight"] = weights
    quarter = f"{start.year}Q{((start.month - 1)//3)+1}"
    test_join["quarter"] = quarter

    long_weights = weights[test_signal == 1]
    short_weights = weights[test_signal == -1]
    selected_weights = weights[selected]
    receipt = {
        "quarter": quarter,
        "ridge_train_rows": int(len(op_train)),
        "ridge_test_rows": int(len(op_test)),
        "ridge_train_last_timestamp": op_train["timestamp"].max().isoformat(),
        "ridge_oof": ridge_oof,
        "risk_train_rows": int(len(risk_train)),
        "risk_train_last_timestamp": risk_train["timestamp"].max().isoformat(),
        "side_training_median_predicted_mae": medians,
        "training_raw_weight_mean_normalization": normalization,
        "signals": int(np.sum(selected)),
        "long_signals": int(np.sum(test_signal == 1)),
        "short_signals": int(np.sum(test_signal == -1)),
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


def phase_audit_weighted(ts: pd.Series, signal: np.ndarray, point_move: np.ndarray, weight: np.ndarray, horizon: int) -> dict:
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    slots = ((parsed - epoch) // pd.Timedelta(minutes=12)).to_numpy(dtype=np.int64)
    phases = {}
    for phase in range(horizon):
        mask = (slots % horizon) == phase
        sig = signal[mask]
        move = point_move[mask]
        w = weight[mask]
        selected = sig != 0
        side = np.where(sig[selected] == 1, 1.0, -1.0)
        gross = side * move[selected] * w[selected]
        rec = {"signals": int(len(gross)), "mean_exposure_weight": float(np.mean(w[selected])) if np.any(selected) else None}
        if len(gross) >= 10:
            rec["gross_mean_weighted_points"] = float(np.mean(gross))
            for cost in POINT_COSTS:
                key = str(cost).replace(".", "p")
                net = gross - float(cost) * w[selected]
                rec[f"net_mean_points_after_{key}pt"] = float(np.mean(net))
        phases[str(phase)] = rec
    valid = [v for v in phases.values() if "net_mean_points_after_1p0pt" in v]
    out = {
        "valid_phases": int(len(valid)),
        "phase_streams": phases,
        "contract": f"absolute UTC 12min slot modulo H{horizon}; same Ridge signals, fractional exposure only",
    }
    if valid:
        for cost in POINT_COSTS:
            key = str(cost).replace(".", "p")
            arr = np.asarray([float(v[f"net_mean_points_after_{key}pt"]) for v in valid], dtype=float)
            out[f"median_phase_net_points_after_{key}pt"] = float(np.median(arr))
            out[f"mean_phase_net_points_after_{key}pt"] = float(np.mean(arr))
            out[f"positive_phase_fraction_after_{key}pt"] = float(np.mean(arr > 0))
    return out


def weekly_rows(joined: pd.DataFrame) -> list[dict]:
    rows = []
    for week_key, group in joined.groupby("trade_week", sort=True):
        if len(group) < 300:
            continue
        signal = group["baseline_signal"].to_numpy(int)
        move = group["point_move"].to_numpy(float)
        budget_weight = group["risk_budget_weight"].to_numpy(float)
        unit_weight = np.where(signal != 0, 1.0, 0.0)
        selected = signal != 0
        rows.append({
            "trade_week": pd.Timestamp(week_key).isoformat(),
            "year": str(pd.Timestamp(week_key).year),
            "quarter": group["quarter"].iloc[0],
            "rows": int(len(group)),
            "baseline": {"phase_audit": phase_audit_weighted(group["timestamp"], signal, move, unit_weight, RIDGE_HORIZON)},
            "risk_budget": {"phase_audit": phase_audit_weighted(group["timestamp"], signal, move, budget_weight, RIDGE_HORIZON)},
            "exposure": {
                "signals": int(np.sum(selected)),
                "mean_weight": float(np.mean(budget_weight[selected])) if np.any(selected) else None,
                "median_weight": float(np.median(budget_weight[selected])) if np.any(selected) else None,
            },
        })
    return rows


def tail_stats(arr: np.ndarray) -> dict:
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


def paired_summary(rows: list[dict], cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    baseline, budget, labels = [], [], []
    for row in rows:
        a = row["baseline"]["phase_audit"].get(field)
        b = row["risk_budget"]["phase_audit"].get(field)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            baseline.append(float(a)); budget.append(float(b)); labels.append(row["trade_week"])
    a = np.asarray(baseline, dtype=float); b = np.asarray(budget, dtype=float); delta = b - a
    if len(delta) == 0:
        return {"weeks": 0, "eligible": False}
    return {
        "weeks": int(len(delta)),
        "eligible": len(delta) >= MIN_PAIRED_WEEKS,
        "baseline": tail_stats(a),
        "risk_budget": tail_stats(b),
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


def exposure_summary(fits: list[dict]) -> dict:
    signals = sum(int(row["signals"]) for row in fits)
    weight_sum = sum(float(row["weight"]["sum"]) for row in fits)
    long_signals = sum(int(row["long_signals"]) for row in fits)
    short_signals = sum(int(row["short_signals"]) for row in fits)
    long_sum = sum(float(row["long_weight_mean"] or 0.0) * int(row["long_signals"]) for row in fits)
    short_sum = sum(float(row["short_weight_mean"] or 0.0) * int(row["short_signals"]) for row in fits)
    return {
        "signals": signals,
        "mean_weight": weight_sum / signals if signals else None,
        "long_signals": long_signals,
        "short_signals": short_signals,
        "long_mean_weight": long_sum / long_signals if long_signals else None,
        "short_mean_weight": short_sum / short_signals if short_signals else None,
        "min_allowed": WEIGHT_MIN,
        "max_allowed": WEIGHT_MAX,
    }


def advance_checks(summary: dict, exposure: dict) -> dict:
    if not summary.get("eligible"):
        checks = {"paired_weeks_at_least_80": False}
    else:
        a = summary["baseline"]
        b = summary["risk_budget"]
        d = summary["delta"]
        baseline_dd = float(a["max_drawdown_points"])
        budget_dd = float(b["max_drawdown_points"])
        checks = {
            "paired_weeks_at_least_80": summary["weeks"] >= MIN_PAIRED_WEEKS,
            "paired_median_delta_nonnegative": d["median"] >= 0,
            "paired_mean_delta_positive": d["mean"] > 0,
            "paired_win_fraction_gt_0p50": d["win_fraction"] > 0.50,
            "positive_week_fraction_no_worse": b["positive_week_fraction"] >= a["positive_week_fraction"],
            "p10_no_worse": b["p10"] >= a["p10"],
            "bottom10pct_no_worse": b["bottom10pct_mean"] >= a["bottom10pct_mean"],
            "max_drawdown_improves_10pct": budget_dd >= 0.90 * baseline_dd,
            "mean_exposure_0p90_to_1p10": exposure.get("mean_weight") is not None and 0.90 <= exposure["mean_weight"] <= 1.10,
        }
    return {"passed": all(checks.values()), "checks": checks, "failed": [k for k, v in checks.items() if not v]}


def decomposition(rows: list[dict], cost: float, group_field: str) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    groups: dict[str, list[float]] = {}
    for row in rows:
        a = row["baseline"]["phase_audit"].get(field)
        b = row["risk_budget"]["phase_audit"].get(field)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            groups.setdefault(str(row[group_field]), []).append(float(b) - float(a))
    return {
        label: {
            "weeks": int(len(vals)),
            "median_delta": float(np.median(vals)),
            "mean_delta": float(np.mean(vals)),
            "win_fraction": float(np.mean(np.asarray(vals, dtype=float) > 0)),
        }
        for label, vals in sorted(groups.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--risk-prereq-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prereq = verify_prerequisite(args.risk_prereq_receipt)
    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    op = opportunity_matrix(stitched)
    risk = build_risk_matrix(bars_1h(stitched))

    quarter_starts = list(pd.date_range(DISCOVERY_START, DISCOVERY_END, freq="QS", tz="UTC"))
    rows, fits = [], []
    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        fitted = fit_quarter(op, risk, start, end)
        if fitted is None:
            continue
        joined, receipt = fitted
        rows.extend(weekly_rows(joined)); fits.append(receipt)
        print("QUARTER", receipt["quarter"], json.dumps(receipt, sort_keys=True))

    by_week: dict[str, dict] = {}
    for row in rows:
        key = row["trade_week"]
        if key not in by_week or row["rows"] > by_week[key]["rows"]:
            by_week[key] = row
    rows = [by_week[key] for key in sorted(by_week)]

    paired = {str(cost): paired_summary(rows, cost) for cost in POINT_COSTS}
    exposure = exposure_summary(fits)
    checks = {"1.0": advance_checks(paired["1.0"], exposure), "2.0": advance_checks(paired["2.0"], exposure)}
    passed = checks["1.0"]["passed"] and checks["2.0"]["passed"]

    result = {
        "schema": "foundry.mnq_ridge_h1_risk_budget_discovery.v1",
        "research_only": True,
        "promotion_authority": False,
        "contract": "research/mnq_ridge_h1_risk_budget_contract_20260901.json",
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "deep_timestamp_contract": contract_receipt(),
        "risk_prerequisite_commit": RISK_PREREQ_COMMIT,
        "risk_prerequisite": prereq,
        "consumer": "inverse_mae_exposure_neutral_budget",
        "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
        "discovery_period": {"start": DISCOVERY_START.isoformat(), "end_exclusive": DISCOVERY_END.isoformat()},
        "fit_receipts": fits,
        "weekly_rows": rows,
        "paired_summary": paired,
        "exposure": exposure,
        "advance_checks": checks,
        "decision": "advance_unchanged_to_corrected_2026_confirmation" if passed else "reject_as_specified",
        "decomposition_after_1pt": {
            "year": decomposition(rows, 1.0, "year"),
            "quarter": decomposition(rows, 1.0, "quarter"),
        },
        "decomposition_after_2pt": {
            "year": decomposition(rows, 2.0, "year"),
            "quarter": decomposition(rows, 2.0, "quarter"),
        },
        "nq_boundary": "NQ remains excluded until an unchanged composite survives MNQ discovery and corrected-2026 confirmation.",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_RIDGE_H1_RISK_BUDGET_DISCOVERY=PASS")
    print("DECISION=" + result["decision"])
    print("EXPOSURE=" + json.dumps(exposure, sort_keys=True))
    print("CHECKS=" + json.dumps(checks, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
