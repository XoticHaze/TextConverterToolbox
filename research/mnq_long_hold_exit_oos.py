from __future__ import annotations

"""Chronological OOS evaluation for the MNQ long HOLD/EXIT specialist.

Research-only. This module evaluates hypothetical already-open long positions. It
cannot authorize runtime exits, StrategySpec changes, broker actions, or model
promotion.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.mnq_long_hold_exit_targets import LongHoldExitSpec, assert_feature_fence


@dataclass(frozen=True)
class HoldExitOOSSpec:
    min_train_rows: int = 200
    ridge: float = 1e-3
    trailing_points: float = 8.0
    atr_multiple: float = 2.0
    probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.min_train_rows < 2:
            raise ValueError("min_train_rows must be >= 2")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if self.trailing_points <= 0:
            raise ValueError("trailing_points must be positive")
        if self.atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive")
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError("probability_threshold must be inside (0, 1)")


def _fit_ridge_probability(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef, mean, scale


def _predict_probability(row: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    coef, mean, scale = fit
    z = (row - mean) / scale
    linear = float(np.r_[1.0, z] @ coef)
    # Ridge probability is deliberately a simple interpretable linear baseline.
    return float(np.clip(linear, 0.0, 1.0))


def chronological_long_hold_exit_panel(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    feature_columns: list[str],
    target_spec: LongHoldExitSpec,
    oos_spec: HoldExitOOSSpec,
    *,
    atr_column: str = "atr_points",
) -> pd.DataFrame:
    """Build paired OOS HOLD/EXIT actions on identical hypothetical long paths.

    The learned linear challenger is refit for every test observation using only
    earlier rows whose forward target horizon has already completed by the
    current decision timestamp. This positional purge prevents unresolved future
    outcomes from entering training.

    Baselines are task-specific and causal at the decision bar:
    * ``fixed_horizon`` always HOLDs to the target horizon;
    * ``trailing`` HOLDs only while decision close remains above the running
      decision-window close maximum minus a fixed point trail;
    * ``atr_trailing`` uses entry-time ATR instead of a fixed trail.

    All policies are evaluated against the same ``pnl_if_exit_points`` and
    ``pnl_if_hold_points`` target columns. Hard deterministic risk exits remain
    outside this research panel and retain downstream veto authority.
    """
    if len(frame) != len(targets) or not frame.index.equals(targets.index):
        raise ValueError("frame and targets must have identical indexed rows")
    assert_feature_fence(feature_columns)
    missing = [c for c in feature_columns + ["close", atr_column] if c not in frame]
    if missing:
        raise ValueError(f"missing causal inputs: {missing}")
    required_targets = {"hold_label", "pnl_if_exit_points", "pnl_if_hold_points"}
    if not required_targets.issubset(targets.columns):
        raise ValueError(f"targets missing required columns: {sorted(required_targets - set(targets.columns))}")

    x = frame[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=float)
    atr = pd.to_numeric(frame[atr_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(close).all() or not np.isfinite(atr).all():
        raise ValueError("causal inputs contain non-finite values")
    if (atr <= 0).any():
        raise ValueError("ATR inputs must be positive")

    labels = targets["hold_label"]
    out_rows: list[dict[str, object]] = []
    n = len(frame)

    for i in range(n):
        if pd.isna(labels.iloc[i]):
            continue
        decision_i = i + target_spec.decision_bars
        horizon_i = i + target_spec.horizon_bars
        if horizon_i >= n or decision_i >= n:
            continue

        # A training row j is eligible only if j's outcome is known by this
        # test row's decision time: j + horizon <= i + decision.
        last_train = decision_i - target_spec.horizon_bars
        eligible = np.arange(max(0, last_train + 1), dtype=int)
        eligible = eligible[~labels.iloc[eligible].isna().to_numpy()]
        if len(eligible) < oos_spec.min_train_rows:
            continue

        y_train = labels.iloc[eligible].astype(float).to_numpy()
        fit = _fit_ridge_probability(x[eligible], y_train, oos_spec.ridge)
        probability = _predict_probability(x[i], fit)
        learned_hold = probability >= oos_spec.probability_threshold

        decision_window = close[i : decision_i + 1]
        running_max = float(np.max(decision_window))
        decision_close = float(close[decision_i])
        trailing_hold = decision_close > running_max - oos_spec.trailing_points
        atr_trailing_hold = decision_close > running_max - oos_spec.atr_multiple * float(atr[i])

        exit_pnl = float(targets["pnl_if_exit_points"].iloc[i])
        hold_pnl = float(targets["pnl_if_hold_points"].iloc[i])

        def reward(hold: bool) -> float:
            return hold_pnl if hold else exit_pnl

        row: dict[str, object] = {
            "row_index": i,
            "train_rows": int(len(eligible)),
            "train_last_row": int(eligible[-1]),
            "decision_row": int(decision_i),
            "target_resolution_row": int(horizon_i),
            "realized_hold_label": int(labels.iloc[i]),
            "learned_hold_probability": probability,
            "learned_action": "HOLD" if learned_hold else "EXIT",
            "fixed_horizon_action": "HOLD",
            "trailing_action": "HOLD" if trailing_hold else "EXIT",
            "atr_trailing_action": "HOLD" if atr_trailing_hold else "EXIT",
            "pnl_if_exit_points": exit_pnl,
            "pnl_if_hold_points": hold_pnl,
            "learned_realized_points": reward(learned_hold),
            "fixed_horizon_realized_points": hold_pnl,
            "trailing_realized_points": reward(trailing_hold),
            "atr_trailing_realized_points": reward(atr_trailing_hold),
            "cost_points": float(target_spec.cost_points),
            "research_only": True,
        }
        if "timestamp" in frame:
            row["timestamp"] = frame["timestamp"].iloc[i]
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def summarize_hold_exit_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return paired economic summaries without selecting a winner."""
    policy_columns = {
        "learned": "learned_realized_points",
        "fixed_horizon": "fixed_horizon_realized_points",
        "trailing": "trailing_realized_points",
        "atr_trailing": "atr_trailing_realized_points",
    }
    rows: list[dict[str, object]] = []
    for policy, column in policy_columns.items():
        if column not in panel:
            raise ValueError(f"panel missing {column}")
        values = pd.to_numeric(panel[column], errors="raise").to_numpy(dtype=float)
        if len(values) == 0:
            rows.append({"policy": policy, "rows": 0, "mean_points": np.nan, "median_points": np.nan, "positive_fraction": np.nan})
            continue
        rows.append(
            {
                "policy": policy,
                "rows": int(len(values)),
                "mean_points": float(np.mean(values)),
                "median_points": float(np.median(values)),
                "positive_fraction": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)
