from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import research.mnq_ridge_h1_mae_orchestration as veto_base
import research.mnq_ridge_h1_risk_budget as budget_base
import research.mnq_ridge_h1_opportunity_risk as ratio_base
from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import deep_bars

CONTRACT = "research/mnq_multitf_rollsafe_revalidation_contract_20260901.json"
CONSUMERS = ("binary_q80_mae_veto", "pure_inverse_mae_budget", "opportunity_to_mae_budget")


def opportunity_matrix_with_contract(stitched: pd.DataFrame) -> pd.DataFrame:
    features = list(BASE_FEATURES)
    work = _add_features(deep_bars(stitched))
    needed = list(dict.fromkeys(["timestamp", "source_contract", "close", "rv_120", *features]))
    work = work[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    work["point_move"] = work["close"].shift(-veto_base.RIDGE_HORIZON) - work["close"]
    scale = (
        work["close"].astype(float)
        * work["rv_120"].astype(float)
        * math.sqrt(veto_base.RIDGE_HORIZON)
    )
    work["target_move_z"] = work["point_move"] / scale.replace(0, np.nan)
    work["trade_week"] = veto_base.trade_week_key(work["timestamp"])
    return work


def contract_safe_join(op_frame: pd.DataFrame, risk_scores: pd.DataFrame) -> pd.DataFrame:
    left = op_frame.copy().reset_index(drop=True)
    left["_row_id"] = np.arange(len(left), dtype=np.int64)
    parts: list[pd.DataFrame] = []
    pred_cols = ["pred_long_mae_z", "pred_short_mae_z"]
    for contract, og in left.groupby("source_contract", sort=False):
        rg = risk_scores[risk_scores["source_contract"] == contract].copy()
        if rg.empty:
            joined = og.copy()
            joined["available_at"] = pd.NaT
            joined["risk_source_contract"] = None
            for col in pred_cols:
                joined[col] = np.nan
        else:
            right = rg[["available_at", "source_contract", *pred_cols]].rename(
                columns={"source_contract": "risk_source_contract"}
            )
            joined = pd.merge_asof(
                og.sort_values("timestamp"),
                right.sort_values("available_at"),
                left_on="timestamp",
                right_on="available_at",
                direction="backward",
                allow_exact_matches=True,
            )
        parts.append(joined)
    out = pd.concat(parts, ignore_index=True).sort_values("_row_id").reset_index(drop=True)
    out["risk_available"] = out[pred_cols].notna().all(axis=1)
    available = out["risk_available"]
    if available.any():
        mismatch = (
            out.loc[available, "risk_source_contract"].astype(str)
            != out.loc[available, "source_contract"].astype(str)
        )
        if mismatch.any():
            raise RuntimeError("contract-safe join produced a cross-contract risk assignment")
        if (out.loc[available, "available_at"] > out.loc[available, "timestamp"]).any():
            raise RuntimeError("contract-safe join produced 1H risk lookahead")
    out["risk_age_minutes"] = np.nan
    out.loc[available, "risk_age_minutes"] = (
        out.loc[available, "timestamp"] - out.loc[available, "available_at"]
    ) / pd.Timedelta(minutes=1)
    return out.drop(columns=["_row_id"])


def legacy_cross_contract_count(op_frame: pd.DataFrame, risk_scores: pd.DataFrame, signal: np.ndarray) -> dict:
    right = risk_scores[
        ["available_at", "source_contract", "pred_long_mae_z", "pred_short_mae_z"]
    ].rename(columns={"source_contract": "risk_source_contract"})
    legacy = pd.merge_asof(
        op_frame.sort_values("timestamp").reset_index(drop=True),
        right.sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    available = legacy[["pred_long_mae_z", "pred_short_mae_z"]].notna().all(axis=1)
    cross = available & (
        legacy["risk_source_contract"].astype(str) != legacy["source_contract"].astype(str)
    )
    sig = np.asarray(signal, dtype=int)
    if len(sig) != len(legacy):
        raise RuntimeError("legacy cross-contract audit signal alignment mismatch")
    return {
        "rows": int(len(legacy)),
        "risk_available_rows": int(available.sum()),
        "cross_contract_rows": int(cross.sum()),
        "selected_ridge_signals": int(np.sum(sig != 0)),
        "cross_contract_selected_signals": int(np.sum(cross.to_numpy() & (sig != 0))),
    }


def weight_summary(weights: np.ndarray, signal: np.ndarray) -> dict:
    selected = signal != 0
    long_mask = signal == 1
    short_mask = signal == -1
    vals = weights[selected]
    return {
        "signals": int(selected.sum()),
        "sum": float(np.sum(vals)),
        "mean": float(np.mean(vals)) if len(vals) else None,
        "median": float(np.median(vals)) if len(vals) else None,
        "p10": float(np.quantile(vals, 0.10)) if len(vals) else None,
        "p90": float(np.quantile(vals, 0.90)) if len(vals) else None,
        "min": float(np.min(vals)) if len(vals) else None,
        "max": float(np.max(vals)) if len(vals) else None,
        "long_signals": int(long_mask.sum()),
        "short_signals": int(short_mask.sum()),
        "long_sum": float(np.sum(weights[long_mask])),
        "short_sum": float(np.sum(weights[short_mask])),
        "long_mean": float(np.mean(weights[long_mask])) if long_mask.any() else None,
        "short_mean": float(np.mean(weights[short_mask])) if short_mask.any() else None,
    }


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
    train_end = first_test - veto_base.RIDGE_HORIZON
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
        raise RuntimeError("roll-safe Ridge chronology violation")

    x_train = op_train[features].to_numpy(float)
    y_train = op_train["target_move_z"].to_numpy(float)
    ridge_threshold, ridge_oof = veto_base.chronological_oof_threshold(
        "ridge", x_train, y_train, veto_base.RIDGE_HORIZON
    )
    ridge = veto_base.make_regressor("ridge").fit(x_train, y_train)
    train_pred_z = np.asarray(ridge.predict(x_train), dtype=float)
    test_pred_z = np.asarray(ridge.predict(op_test[features].to_numpy(float)), dtype=float)
    train_signal = np.where(
        np.abs(train_pred_z) >= ridge_threshold, np.sign(train_pred_z), 0
    ).astype(int)
    test_signal = np.where(
        np.abs(test_pred_z) >= ridge_threshold, np.sign(test_pred_z), 0
    ).astype(int)

    risk_train = risk[
        risk["timestamp"] < start - pd.Timedelta(hours=veto_base.RISK_HORIZON)
    ].dropna(subset=list(veto_base.RISK_TARGETS)).copy()
    if len(risk_train) < 5000:
        return None
    rx = risk_train[veto_base.RISK_FEATURES].to_numpy(float)
    risk_scores = risk.copy()
    sx = risk_scores[veto_base.RISK_FEATURES].to_numpy(float)
    veto_thresholds: dict[str, float] = {}
    for target in veto_base.RISK_TARGETS:
        model = veto_base.risk_model().fit(rx, risk_train[target].to_numpy(float))
        train_risk_pred = np.asarray(model.predict(rx), dtype=float)
        veto_thresholds[target] = float(np.quantile(train_risk_pred, veto_base.RISK_QUANTILE))
        risk_scores[f"pred_{target}"] = model.predict(sx)

    train_for_join = op_train.copy()
    train_for_join["_ridge_signal"] = train_signal
    train_for_join["_ridge_abs_pred_z"] = np.abs(train_pred_z)
    train_join = contract_safe_join(train_for_join, risk_scores)

    test_for_join = op_test.copy()
    test_for_join["_ridge_signal"] = test_signal
    test_for_join["_ridge_abs_pred_z"] = np.abs(test_pred_z)
    joined = contract_safe_join(test_for_join, risk_scores)
    if len(joined) != len(op_test):
        raise RuntimeError("roll-safe join changed OOS Ridge row count")

    available_train = train_join["risk_available"].to_numpy(bool)
    available_test = joined["risk_available"].to_numpy(bool)
    train_join_signal = train_join["_ridge_signal"].to_numpy(int)
    train_join_abs_pred = train_join["_ridge_abs_pred_z"].to_numpy(float)
    joined_signal = joined["_ridge_signal"].to_numpy(int)
    joined_abs_pred = joined["_ridge_abs_pred_z"].to_numpy(float)
    if not np.array_equal(joined_signal, test_signal):
        raise RuntimeError("roll-safe join changed Ridge signal alignment")

    pure_medians: dict[str, float] = {}
    ratio_refs: dict[str, dict[str, float]] = {}
    for side, label, target in (
        (1, "long", "long_mae_z"),
        (-1, "short", "short_mae_z"),
    ):
        mask = available_train & (train_join_signal == side)
        risk_vals = train_join.loc[mask, f"pred_{target}"].to_numpy(float)
        opp_vals = train_join_abs_pred[mask]
        if len(risk_vals) < 1000:
            raise RuntimeError(
                f"insufficient same-contract prior {label} Ridge signals for roll-safe references: {len(risk_vals)}"
            )
        pure_medians[target] = float(np.median(risk_vals))
        ratio_refs[label] = {
            "signals": int(len(risk_vals)),
            "median_abs_ridge_pred_z": float(np.median(opp_vals)),
            "median_predicted_mae_z": float(np.median(risk_vals)),
        }

    train_ref_mask = available_train
    ref_signal = train_join_signal[train_ref_mask]
    pure_train_raw = budget_base.side_raw_weight(
        ref_signal,
        train_join.loc[train_ref_mask, "pred_long_mae_z"].to_numpy(float),
        train_join.loc[train_ref_mask, "pred_short_mae_z"].to_numpy(float),
        pure_medians,
    )
    pure_selected = ref_signal != 0
    pure_norm = float(np.mean(pure_train_raw[pure_selected]))
    if not np.isfinite(pure_norm) or pure_norm <= 0:
        raise RuntimeError("invalid roll-safe pure-risk normalization")

    ratio_train_raw = ratio_base.opportunity_risk_raw_weight(
        ref_signal,
        train_join_abs_pred[train_ref_mask],
        train_join.loc[train_ref_mask, "pred_long_mae_z"].to_numpy(float),
        train_join.loc[train_ref_mask, "pred_short_mae_z"].to_numpy(float),
        ratio_refs,
    )
    ratio_selected = ref_signal != 0
    ratio_norm = float(np.mean(ratio_train_raw[ratio_selected]))
    if not np.isfinite(ratio_norm) or ratio_norm <= 0:
        raise RuntimeError("invalid roll-safe opportunity-risk normalization")

    veto_signal = test_signal.copy()
    valid_long = available_test & (test_signal == 1)
    valid_short = available_test & (test_signal == -1)
    long_veto = valid_long & (
        joined["pred_long_mae_z"].fillna(-np.inf).to_numpy(float)
        >= veto_thresholds["long_mae_z"]
    )
    short_veto = valid_short & (
        joined["pred_short_mae_z"].fillna(-np.inf).to_numpy(float)
        >= veto_thresholds["short_mae_z"]
    )
    veto_signal[long_veto | short_veto] = 0

    pure_weight = np.where(test_signal != 0, 1.0, 0.0)
    ratio_weight = np.where(test_signal != 0, 1.0, 0.0)
    valid_selected = available_test & (test_signal != 0)
    if valid_selected.any():
        sub_signal = test_signal[valid_selected]
        pure_raw = budget_base.side_raw_weight(
            sub_signal,
            joined.loc[valid_selected, "pred_long_mae_z"].to_numpy(float),
            joined.loc[valid_selected, "pred_short_mae_z"].to_numpy(float),
            pure_medians,
        )
        pure_weight[valid_selected] = np.clip(
            pure_raw / pure_norm, budget_base.WEIGHT_MIN, budget_base.WEIGHT_MAX
        )
        ratio_raw = ratio_base.opportunity_risk_raw_weight(
            sub_signal,
            joined_abs_pred[valid_selected],
            joined.loc[valid_selected, "pred_long_mae_z"].to_numpy(float),
            joined.loc[valid_selected, "pred_short_mae_z"].to_numpy(float),
            ratio_refs,
        )
        ratio_weight[valid_selected] = np.clip(
            ratio_raw / ratio_norm, budget_base.WEIGHT_MIN, budget_base.WEIGHT_MAX
        )

    joined["baseline_signal"] = test_signal
    joined["veto_signal"] = veto_signal
    joined["pure_inverse_mae_weight"] = pure_weight
    joined["opportunity_to_mae_weight"] = ratio_weight
    joined["quarter"] = f"{start.year}Q{((start.month - 1) // 3) + 1}"

    legacy = legacy_cross_contract_count(op_test, risk_scores, test_signal)
    valid_age = joined.loc[available_test, "risk_age_minutes"].dropna()
    selected_mask = test_signal != 0
    selected_available = available_test & selected_mask
    selected_unavailable = (~available_test) & selected_mask

    receipt = {
        "quarter": joined["quarter"].iloc[0],
        "ridge_train_rows": int(len(op_train)),
        "ridge_test_rows": int(len(op_test)),
        "ridge_train_last_timestamp": op_train["timestamp"].max().isoformat(),
        "ridge_oof": ridge_oof,
        "risk_train_rows": int(len(risk_train)),
        "risk_train_last_timestamp": risk_train["timestamp"].max().isoformat(),
        "veto_thresholds": veto_thresholds,
        "pure_training_median_predicted_mae": pure_medians,
        "pure_training_normalization": pure_norm,
        "ratio_training_references": ratio_refs,
        "ratio_training_normalization": ratio_norm,
        "signals": int(selected_mask.sum()),
        "long_signals": int(np.sum(test_signal == 1)),
        "short_signals": int(np.sum(test_signal == -1)),
        "same_contract_risk_available_rows": int(available_test.sum()),
        "same_contract_risk_unavailable_rows": int((~available_test).sum()),
        "same_contract_risk_available_selected_signals": int(selected_available.sum()),
        "same_contract_risk_unavailable_selected_signals": int(selected_unavailable.sum()),
        "same_contract_non_neutral_source_mismatches": 0,
        "legacy_join_audit": legacy,
        "same_contract_risk_age_minutes": {
            "median": float(valid_age.median()) if len(valid_age) else None,
            "p95": float(valid_age.quantile(0.95)) if len(valid_age) else None,
            "max": float(valid_age.max()) if len(valid_age) else None,
        },
        "veto": {
            "long_vetoes": int(long_veto.sum()),
            "short_vetoes": int(short_veto.sum()),
            "neutral_unavailable_selected_signals": int(selected_unavailable.sum()),
        },
        "pure_weight": weight_summary(pure_weight, test_signal),
        "ratio_weight": weight_summary(ratio_weight, test_signal),
    }
    return joined, receipt


def weekly_rows(joined: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for week_key, g in joined.groupby("trade_week", sort=True):
        if len(g) < 300:
            continue
        base_signal = g["baseline_signal"].to_numpy(int)
        veto_signal = g["veto_signal"].to_numpy(int)
        move = g["point_move"].to_numpy(float)
        base_weight = np.where(base_signal != 0, 1.0, 0.0)
        veto_weight = np.where(veto_signal != 0, 1.0, 0.0)
        rows.append({
            "trade_week": pd.Timestamp(week_key).isoformat(),
            "year": str(pd.Timestamp(week_key).year),
            "quarter": str(g["quarter"].iloc[0]),
            "rows": int(len(g)),
            "baseline": {
                "phase_audit": budget_base.phase_audit_weighted(
                    g["timestamp"], base_signal, move, base_weight, veto_base.RIDGE_HORIZON
                )
            },
            "binary_q80_mae_veto": {
                "phase_audit": budget_base.phase_audit_weighted(
                    g["timestamp"], veto_signal, move, veto_weight, veto_base.RIDGE_HORIZON
                )
            },
            "pure_inverse_mae_budget": {
                "phase_audit": budget_base.phase_audit_weighted(
                    g["timestamp"],
                    base_signal,
                    move,
                    g["pure_inverse_mae_weight"].to_numpy(float),
                    veto_base.RIDGE_HORIZON,
                )
            },
            "opportunity_to_mae_budget": {
                "phase_audit": budget_base.phase_audit_weighted(
                    g["timestamp"],
                    base_signal,
                    move,
                    g["opportunity_to_mae_weight"].to_numpy(float),
                    veto_base.RIDGE_HORIZON,
                )
            },
        })
    return rows


def paired_summary(rows: list[dict], policy: str, cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    baseline: list[float] = []
    candidate: list[float] = []
    labels: list[str] = []
    for row in rows:
        b = row["baseline"]["phase_audit"].get(field)
        c = row[policy]["phase_audit"].get(field)
        if b is None or c is None or not np.isfinite(b) or not np.isfinite(c):
            continue
        baseline.append(float(b))
        candidate.append(float(c))
        labels.append(row["trade_week"])
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(candidate, dtype=float)
    delta = c - b
    if len(delta) == 0:
        return {"weeks": 0, "eligible": False}
    return {
        "weeks": int(len(delta)),
        "eligible": len(delta) >= veto_base.MIN_PAIRED_WEEKS,
        "baseline": budget_base.tail_stats(b),
        "candidate": budget_base.tail_stats(c),
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


def veto_checks(summary: dict) -> dict:
    if not summary.get("eligible"):
        checks = {"paired_weeks_at_least_80": False}
    else:
        b, c, d = summary["baseline"], summary["candidate"], summary["delta"]
        checks = {
            "paired_weeks_at_least_80": summary["weeks"] >= veto_base.MIN_PAIRED_WEEKS,
            "positive_paired_median_delta": d["median"] > 0,
            "positive_paired_mean_delta": d["mean"] > 0,
            "paired_win_fraction_gt_0p50": d["win_fraction"] > 0.50,
            "conditioned_positive_week_fraction_ge_baseline": c["positive_week_fraction"] >= b["positive_week_fraction"],
            "conditioned_p10_ge_baseline": c["p10"] >= b["p10"],
            "conditioned_bottom10pct_ge_baseline": c["bottom10pct_mean"] >= b["bottom10pct_mean"],
            "conditioned_drawdown_no_worse": c["max_drawdown_points"] >= b["max_drawdown_points"],
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def budget_checks(summary: dict, exposure: dict) -> dict:
    if not summary.get("eligible"):
        checks = {"paired_weeks_at_least_80": False}
    else:
        b, c, d = summary["baseline"], summary["candidate"], summary["delta"]
        baseline_dd = float(b["max_drawdown_points"])
        candidate_dd = float(c["max_drawdown_points"])
        if baseline_dd < 0:
            dd_improvement = (candidate_dd - baseline_dd) / abs(baseline_dd)
        else:
            dd_improvement = 0.0
        checks = {
            "paired_weeks_at_least_80": summary["weeks"] >= veto_base.MIN_PAIRED_WEEKS,
            "paired_mean_delta_positive": d["mean"] > 0,
            "paired_median_delta_nonnegative": d["median"] >= 0,
            "paired_win_fraction_gt_0p50": d["win_fraction"] > 0.50,
            "positive_week_fraction_no_worse": c["positive_week_fraction"] >= b["positive_week_fraction"],
            "p10_no_worse": c["p10"] >= b["p10"],
            "bottom10pct_no_worse": c["bottom10pct_mean"] >= b["bottom10pct_mean"],
            "max_drawdown_improves_10pct": dd_improvement >= 0.10,
            "mean_exposure_0p90_to_1p10": 0.90 <= float(exposure["mean_weight"]) <= 1.10,
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def exposure_summary(fits: list[dict], key: str) -> dict:
    signals = sum(int(r[key]["signals"]) for r in fits)
    total = sum(float(r[key]["sum"]) for r in fits)
    long_signals = sum(int(r[key]["long_signals"]) for r in fits)
    short_signals = sum(int(r[key]["short_signals"]) for r in fits)
    long_total = sum(float(r[key]["long_sum"]) for r in fits)
    short_total = sum(float(r[key]["short_sum"]) for r in fits)
    return {
        "signals": signals,
        "mean_weight": total / signals if signals else None,
        "long_signals": long_signals,
        "short_signals": short_signals,
        "long_mean_weight": long_total / long_signals if long_signals else None,
        "short_mean_weight": short_total / short_signals if short_signals else None,
        "min_allowed": budget_base.WEIGHT_MIN,
        "max_allowed": budget_base.WEIGHT_MAX,
    }


def decomposition(rows: list[dict], policy: str, cost: float, field: str) -> dict:
    key = str(cost).replace(".", "p")
    metric = f"median_phase_net_points_after_{key}pt"
    groups: dict[str, list[float]] = {}
    for row in rows:
        b = row["baseline"]["phase_audit"].get(metric)
        c = row[policy]["phase_audit"].get(metric)
        if b is None or c is None or not np.isfinite(b) or not np.isfinite(c):
            continue
        groups.setdefault(str(row[field]), []).append(float(c) - float(b))
    return {
        label: {
            "weeks": len(vals),
            "median_delta": float(np.median(vals)),
            "mean_delta": float(np.mean(vals)),
            "win_fraction": float(np.mean(np.asarray(vals) > 0)),
        }
        for label, vals in sorted(groups.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--risk-prereq-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prereq = veto_base.verify_prerequisite(args.risk_prereq_receipt)
    raw = veto_base.load_deep(args.deep_root)
    stitched = veto_base.stitch_deep(raw, veto_base.deep_roll_schedule(raw))
    op = opportunity_matrix_with_contract(stitched)
    risk = veto_base.build_risk_matrix(veto_base.bars_1h(stitched))

    quarter_starts = list(
        pd.date_range(veto_base.DISCOVERY_START, veto_base.DISCOVERY_END, freq="QS", tz="UTC")
    )
    fits: list[dict] = []
    rows: list[dict] = []
    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        fitted = fit_quarter(op, risk, start, end)
        if fitted is None:
            continue
        joined, receipt = fitted
        fits.append(receipt)
        rows.extend(weekly_rows(joined))
        print("QUARTER", receipt["quarter"], json.dumps(receipt, sort_keys=True))

    by_week: dict[str, dict] = {}
    for row in rows:
        key = row["trade_week"]
        if key not in by_week or row["rows"] > by_week[key]["rows"]:
            by_week[key] = row
    rows = [by_week[key] for key in sorted(by_week)]

    summaries = {
        policy: {str(cost): paired_summary(rows, policy, cost) for cost in veto_base.POINT_COSTS}
        for policy in CONSUMERS
    }
    exposures = {
        "pure_inverse_mae_budget": exposure_summary(fits, "pure_weight"),
        "opportunity_to_mae_budget": exposure_summary(fits, "ratio_weight"),
    }
    checks = {
        "binary_q80_mae_veto": {
            "1.0": veto_checks(summaries["binary_q80_mae_veto"]["1.0"]),
            "2.0": veto_checks(summaries["binary_q80_mae_veto"]["2.0"]),
        },
        "pure_inverse_mae_budget": {
            "1.0": budget_checks(summaries["pure_inverse_mae_budget"]["1.0"], exposures["pure_inverse_mae_budget"]),
            "2.0": budget_checks(summaries["pure_inverse_mae_budget"]["2.0"], exposures["pure_inverse_mae_budget"]),
        },
        "opportunity_to_mae_budget": {
            "1.0": budget_checks(summaries["opportunity_to_mae_budget"]["1.0"], exposures["opportunity_to_mae_budget"]),
            "2.0": budget_checks(summaries["opportunity_to_mae_budget"]["2.0"], exposures["opportunity_to_mae_budget"]),
        },
    }
    decisions = {
        policy: (
            "advance_unchanged_to_corrected_2026_confirmation"
            if checks[policy]["1.0"]["passed"] and checks[policy]["2.0"]["passed"]
            else "reject_as_specified_under_roll_safe_join"
        )
        for policy in CONSUMERS
    }

    integrity = {
        "legacy_cross_contract_rows": int(sum(r["legacy_join_audit"]["cross_contract_rows"] for r in fits)),
        "legacy_cross_contract_selected_signals": int(sum(r["legacy_join_audit"]["cross_contract_selected_signals"] for r in fits)),
        "same_contract_risk_unavailable_rows": int(sum(r["same_contract_risk_unavailable_rows"] for r in fits)),
        "same_contract_risk_unavailable_selected_signals": int(sum(r["same_contract_risk_unavailable_selected_signals"] for r in fits)),
        "non_neutral_source_contract_mismatches": int(sum(r["same_contract_non_neutral_source_mismatches"] for r in fits)),
        "by_quarter": {
            r["quarter"]: {
                "legacy_cross_contract_rows": r["legacy_join_audit"]["cross_contract_rows"],
                "legacy_cross_contract_selected_signals": r["legacy_join_audit"]["cross_contract_selected_signals"],
                "same_contract_risk_unavailable_rows": r["same_contract_risk_unavailable_rows"],
                "same_contract_risk_unavailable_selected_signals": r["same_contract_risk_unavailable_selected_signals"],
                "same_contract_risk_age_minutes": r["same_contract_risk_age_minutes"],
            }
            for r in fits
        },
    }
    if integrity["non_neutral_source_contract_mismatches"] != 0:
        raise RuntimeError("roll-safe revalidation retained cross-contract non-neutral risk assignments")

    result = {
        "schema": "foundry.mnq_multitf_rollsafe_revalidation.v1",
        "research_only": True,
        "promotion_authority": False,
        "contract": CONTRACT,
        "source": f"mbytes21/MNQ_DATA@{veto_base.SOURCE_COMMIT}",
        "deep_timestamp_contract": veto_base.contract_receipt(),
        "risk_prerequisite_commit": veto_base.RISK_PREREQ_COMMIT,
        "risk_prerequisite": prereq,
        "roll_safe_semantics": "same source contract only; missing same-contract warmed risk is neutral/no-op for all risk consumers",
        "fit_receipts": fits,
        "weekly_rows": rows,
        "paired_summary": summaries,
        "exposure": exposures,
        "advance_checks": checks,
        "decision": decisions,
        "integrity": integrity,
        "decomposition_after_1pt": {
            policy: {
                "year": decomposition(rows, policy, 1.0, "year"),
                "quarter": decomposition(rows, policy, 1.0, "quarter"),
            }
            for policy in CONSUMERS
        },
        "nq_boundary": "NQ excluded from roll-safe integrity revalidation.",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_MULTITF_ROLLSAFE_REVALIDATION=PASS")
    print("INTEGRITY=" + json.dumps(integrity, sort_keys=True))
    print("DECISIONS=" + json.dumps(decisions, sort_keys=True))
    print("CHECKS=" + json.dumps(checks, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
