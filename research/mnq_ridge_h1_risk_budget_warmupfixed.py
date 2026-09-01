from __future__ import annotations

import numpy as np
import pandas as pd

import research.mnq_ridge_h1_risk_budget as base
from research.expanded_regime_ablation import BASE_FEATURES

WARMUP_FIX_GENERATION = "training_availability_r1"


def join_risk_scores_available(
    op_frame: pd.DataFrame,
    risk_scores: pd.DataFrame,
    *,
    strict: bool,
) -> pd.DataFrame:
    """Join only chronology-available completed-hour risk scores.

    Training history may legitimately begin before the 1H feature matrix is
    warmed. Those pre-availability rows are excluded from the risk-budget
    normalization sample only. Once the first 1H risk score exists, gaps fail
    closed. OOS/test joins are always strict.
    """
    joined = pd.merge_asof(
        op_frame.sort_values("timestamp"),
        risk_scores[["available_at", "pred_long_mae_z", "pred_short_mae_z"]].sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    needed = ["pred_long_mae_z", "pred_short_mae_z"]
    missing = joined[needed].isna().any(axis=1)
    if risk_scores.empty or risk_scores["available_at"].dropna().empty:
        raise RuntimeError("no completed-hour 1H risk scores available")
    first_available = risk_scores["available_at"].dropna().min()

    if strict and missing.any():
        raise RuntimeError("missing completed-hour 1H risk score for OOS Ridge decision")

    if not strict and missing.any():
        illegal_gap = missing & (joined["timestamp"] >= first_available)
        if illegal_gap.any():
            raise RuntimeError("missing completed-hour 1H risk score after risk availability began")
        joined = joined.loc[~missing].copy()

    if (joined["available_at"] > joined["timestamp"]).any():
        raise RuntimeError("1H risk lookahead detected")
    joined["risk_age_minutes"] = (joined["timestamp"] - joined["available_at"]) / pd.Timedelta(minutes=1)
    return joined


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
        raise RuntimeError("Ridge risk-budget chronology violation")

    x_train = op_train[features].to_numpy(float)
    y_train = op_train["target_move_z"].to_numpy(float)
    ridge_threshold, ridge_oof = base.chronological_oof_threshold(
        "ridge", x_train, y_train, base.RIDGE_HORIZON
    )
    ridge = base.make_regressor("ridge").fit(x_train, y_train)
    train_pred_z = ridge.predict(x_train)
    test_pred_z = ridge.predict(op_test[features].to_numpy(float))
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
    train_join = join_risk_scores_available(train_for_join, risk_scores, strict=False)
    available_train_signal = train_join.pop("_ridge_signal").to_numpy(int)

    test_for_join = op_test.copy()
    test_for_join["_ridge_signal"] = test_signal
    test_join = join_risk_scores_available(test_for_join, risk_scores, strict=True)
    available_test_signal = test_join.pop("_ridge_signal").to_numpy(int)
    if len(test_join) != len(op_test):
        raise RuntimeError("strict OOS 1H risk availability changed Ridge decision support")

    medians: dict[str, float] = {}
    for side, target in ((1, "long_mae_z"), (-1, "short_mae_z")):
        pred_col = f"pred_{target}"
        vals = train_join.loc[available_train_signal == side, pred_col].to_numpy(float)
        if len(vals) < 1000:
            raise RuntimeError(
                f"insufficient prior Ridge signals for {target} risk-budget reference: {len(vals)}"
            )
        medians[target] = float(np.median(vals))

    train_raw = base.side_raw_weight(
        available_train_signal,
        train_join["pred_long_mae_z"].to_numpy(float),
        train_join["pred_short_mae_z"].to_numpy(float),
        medians,
    )
    train_selected = available_train_signal != 0
    normalization = float(np.mean(train_raw[train_selected]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise RuntimeError("invalid training-only risk-budget normalization")

    test_raw = base.side_raw_weight(
        available_test_signal,
        test_join["pred_long_mae_z"].to_numpy(float),
        test_join["pred_short_mae_z"].to_numpy(float),
        medians,
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

    long_weights = weights[available_test_signal == 1]
    short_weights = weights[available_test_signal == -1]
    selected_weights = weights[selected]
    excluded_train_rows = int(len(op_train) - len(train_join))
    receipt = {
        "quarter": quarter,
        "warmup_fix_generation": WARMUP_FIX_GENERATION,
        "ridge_train_rows": int(len(op_train)),
        "risk_available_ridge_train_rows": int(len(train_join)),
        "risk_warmup_ridge_train_rows_excluded": excluded_train_rows,
        "ridge_test_rows": int(len(op_test)),
        "strict_oos_risk_rows": int(len(test_join)),
        "ridge_train_last_timestamp": op_train["timestamp"].max().isoformat(),
        "ridge_oof": ridge_oof,
        "risk_train_rows": int(len(risk_train)),
        "risk_train_last_timestamp": risk_train["timestamp"].max().isoformat(),
        "side_training_median_predicted_mae": medians,
        "training_raw_weight_mean_normalization": normalization,
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


base.fit_quarter = fit_quarter


if __name__ == "__main__":
    print("RIDGE_H1_RISK_BUDGET_WARMUP_FIX=" + WARMUP_FIX_GENERATION)
    raise SystemExit(base.main())
