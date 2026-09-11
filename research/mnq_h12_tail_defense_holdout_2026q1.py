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
from research.mnq_ridge_h1_mae_orchestration import RIDGE_HORIZON, opportunity_matrix
from research.nq_to_mnq_execution_transfer import POINT_COSTS, phase_audit

CONTRACT = "research/mnq_h12_tail_defense_holdout_2026q1_contract.json"
HOLDOUT_START = pd.Timestamp("2026-01-01T00:00:00Z")
HOLDOUT_END = pd.Timestamp("2026-03-01T00:00:00Z")
MIN_WEEK_ROWS = 300
POLICIES = ("same_selected_always_long", "flat_on_negative", "full_ridge")


def _tail_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    if not len(arr):
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
        "worst": float(np.min(arr)),
        "best": float(np.max(arr)),
        "cumulative_points": float(np.sum(arr)),
        "max_drawdown_points": float(np.min(draw)),
    }


def _weekly_rows(test: pd.DataFrame, policies: dict[str, np.ndarray]) -> list[dict]:
    work = test.copy()
    work["_pos"] = np.arange(len(work))
    rows: list[dict] = []
    for week_key, group in work.groupby("trade_week", sort=True):
        if len(group) < MIN_WEEK_ROWS:
            continue
        pos = group["_pos"].to_numpy(int)
        move = group["point_move"].to_numpy(float)
        node = {}
        for name in POLICIES:
            sig = policies[name][pos]
            node[name] = {
                "coverage": float(np.mean(sig != 0)),
                "long_signals": int(np.sum(sig == 1)),
                "short_signals": int(np.sum(sig == -1)),
                "phase_audit": phase_audit(group["timestamp"], sig, move, RIDGE_HORIZON),
            }
        rows.append({
            "trade_week": pd.Timestamp(week_key).isoformat(),
            "rows": int(len(group)),
            "policies": node,
        })
    return rows


def _weekly_values(rows: list[dict], policy: str, cost: float) -> tuple[np.ndarray, list[str]]:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    values: list[float] = []
    labels: list[str] = []
    for row in rows:
        value = row["policies"][policy]["phase_audit"].get(field)
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))
        labels.append(row["trade_week"])
    return np.asarray(values, dtype=float), labels


def _summary(rows: list[dict], policy: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        values, labels = _weekly_values(rows, policy, cost)
        rec = _tail_stats(values)
        if labels:
            rec["first_week"] = labels[0]
            rec["last_week"] = labels[-1]
        out[str(cost)] = rec
    return out


def _paired(rows: list[dict], challenger: str, control: str, cost: float) -> dict:
    a, labels_a = _weekly_values(rows, challenger, cost)
    b, labels_b = _weekly_values(rows, control, cost)
    if labels_a != labels_b:
        raise RuntimeError("paired week labels differ")
    if not len(a):
        return {"weeks": 0}
    d = a - b
    return {
        "weeks": int(len(d)),
        "challenger": _tail_stats(a),
        "control": _tail_stats(b),
        "delta": {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "win_fraction": float(np.mean(d > 0)),
            "tie_fraction": float(np.mean(d == 0)),
            "worst": float(np.min(d)),
            "best": float(np.max(d)),
        },
        "first_week": labels_a[0],
        "last_week": labels_a[-1],
    }


def _outcome_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"rows": 0}
    return {
        "rows": int(len(x)),
        "mean_point_move": float(np.mean(x)),
        "median_point_move": float(np.median(x)),
        "positive_fraction": float(np.mean(x > 0)),
        "negative_fraction": float(np.mean(x < 0)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    contract = json.loads(Path(CONTRACT).read_text())
    if contract.get("created_before_outcome_inspection") is not True:
        raise RuntimeError("holdout contract was not frozen before outcomes")

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    op = opportunity_matrix(stitched)
    features = list(BASE_FEATURES)

    mask = (
        (op["timestamp"] >= HOLDOUT_START)
        & (op["timestamp"] < HOLDOUT_END)
        & op["target_move_z"].notna()
        & op["point_move"].notna()
    )
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) < 1000:
        raise RuntimeError(f"insufficient resumed 2026Q1 holdout rows: {len(idx)}")
    test_start = int(idx[0])
    train_end = test_start - RIDGE_HORIZON
    if train_end < 50_000:
        raise RuntimeError("insufficient causal training history")
    train = op.iloc[:train_end].copy()
    train = train[train["target_move_z"].notna()].copy()
    test = op.iloc[int(idx[0]) : int(idx[-1] + 1)].copy()
    test = test[
        (test["timestamp"] >= HOLDOUT_START)
        & (test["timestamp"] < HOLDOUT_END)
        & test["target_move_z"].notna()
        & test["point_move"].notna()
    ].copy()
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("holdout chronology violation")

    x_train = train[features].to_numpy(float)
    y_train = train["target_move_z"].to_numpy(float)
    threshold, threshold_authority = chronological_oof_threshold("ridge", x_train, y_train, RIDGE_HORIZON)
    model = make_regressor("ridge").fit(x_train, y_train)
    pred = np.asarray(model.predict(test[features].to_numpy(float)), dtype=float)
    selected = np.abs(pred) >= threshold
    ridge = np.where(selected, np.sign(pred), 0).astype(int)
    policies = {
        "same_selected_always_long": np.where(selected, 1, 0).astype(int),
        "flat_on_negative": np.where(ridge == 1, 1, 0).astype(int),
        "full_ridge": ridge,
    }

    rows = _weekly_rows(test, policies)
    summaries = {name: _summary(rows, name) for name in POLICIES}
    paired = {str(cost): _paired(rows, "full_ridge", "flat_on_negative", cost) for cost in POINT_COSTS}

    checks = {}
    enough = True
    for cost in (1.0, 2.0):
        rec = paired[str(cost)]
        prefix = f"after_{str(cost).replace('.', 'p')}pt"
        checks[f"{prefix}_weeks_ge_4"] = rec.get("weeks", 0) >= 4
        if rec.get("weeks", 0) < 4:
            enough = False
            continue
        checks[f"{prefix}_cumulative_gt_flat"] = rec["challenger"]["cumulative_points"] > rec["control"]["cumulative_points"]
        checks[f"{prefix}_maxdd_better_than_flat"] = rec["challenger"]["max_drawdown_points"] > rec["control"]["max_drawdown_points"]
        checks[f"{prefix}_worst_week_better_than_flat"] = rec["challenger"]["worst"] > rec["control"]["worst"]

    if not enough:
        decision = "H12_TAIL_DEFENSE_2026Q1__INSUFFICIENT_BREADTH"
    elif all(checks.values()):
        decision = "H12_TAIL_DEFENSE_2026Q1__EARLY_HOLDOUT_SUPPORT"
    else:
        decision = "H12_TAIL_DEFENSE_2026Q1__NOT_CONFIRMED"

    moves = test["point_move"].to_numpy(float)
    neg = ridge == -1
    pos = ridge == 1
    result = {
        "schema": "research.mnq_h12_tail_defense_holdout_2026q1.v1",
        "experiment_id": "MNQ_H12_TAIL_DEFENSE_2026Q1_R1",
        "research_only": True,
        "promotion_authority": False,
        "contract": CONTRACT,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "deep_timestamp_contract": contract_receipt(),
        "holdout": {
            "start": HOLDOUT_START.isoformat(),
            "end_exclusive": HOLDOUT_END.isoformat(),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
        },
        "unchanged_ridge": {
            "threshold_abs_pred_z": float(threshold),
            "threshold_authority": threshold_authority,
            "selected_rows": int(np.sum(selected)),
            "long_selected": int(np.sum(pos)),
            "negative_selected": int(np.sum(neg)),
        },
        "selected_outcomes": {
            "positive_ridge_sign": _outcome_stats(moves[pos]),
            "negative_ridge_sign_underlying_long_move": _outcome_stats(moves[neg]),
            "negative_ridge_sign_short_payoff_gross": _outcome_stats(-moves[neg]),
        },
        "weekly_rows": rows,
        "policy_summary": summaries,
        "paired_full_ridge_vs_flat_on_negative": paired,
        "early_holdout_checks": checks,
        "decision": decision,
        "interpretation_boundary": "This short resumed 2026Q1 slice may support or fail to confirm episodic tail-defense. It cannot establish broad symmetric directional alpha, alter the Ridge/q50 policy, or confer trading authority.",
        "boundaries": {
            "strategy_spec_write": False,
            "runtime_activation": False,
            "broker_submit": False,
            "live_trading_change": False,
        },
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("H12_TAIL_DEFENSE_HOLDOUT=" + json.dumps({
        "decision": decision,
        "test_rows": len(test),
        "selected": int(np.sum(selected)),
        "long": int(np.sum(pos)),
        "negative": int(np.sum(neg)),
        "weeks_1pt": paired["1.0"].get("weeks", 0),
        "ridge_1pt": summaries["full_ridge"]["1.0"],
        "flat_1pt": summaries["flat_on_negative"]["1.0"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
