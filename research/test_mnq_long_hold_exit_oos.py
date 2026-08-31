import numpy as np
import pandas as pd
import pytest

from research.mnq_long_hold_exit_oos import (
    HoldExitOOSSpec,
    chronological_long_hold_exit_panel,
    summarize_hold_exit_panel,
)
from research.mnq_long_hold_exit_targets import LongHoldExitSpec, materialize_long_hold_exit_targets


def _frame(n: int = 32) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="12min", tz="UTC"),
            "close": 100.0 + 0.15 * x + np.sin(x / 2.0),
            "momentum": np.sin(x / 3.0),
            "rv": 1.0 + 0.05 * np.cos(x / 4.0),
            "atr_points": 1.5 + 0.1 * np.sin(x / 5.0),
        }
    )


def test_oos_training_is_purged_to_labels_known_by_decision_time():
    frame = _frame()
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2, cost_points=0.5)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    oos_spec = HoldExitOOSSpec(min_train_rows=3, trailing_points=1.0, atr_multiple=2.0)

    panel = chronological_long_hold_exit_panel(
        frame, targets, ["momentum", "rv"], target_spec, oos_spec
    )
    assert not panel.empty
    assert (panel["train_last_row"] + target_spec.horizon_bars <= panel["decision_row"]).all()
    assert (panel["train_last_row"] < panel["row_index"]).all()


def test_future_label_mutation_cannot_change_earliest_oos_prediction():
    frame = _frame()
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2, cost_points=0.5)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    oos_spec = HoldExitOOSSpec(min_train_rows=3)

    original = chronological_long_hold_exit_panel(
        frame, targets, ["momentum", "rv"], target_spec, oos_spec
    )
    first = original.iloc[0]

    tampered = targets.copy()
    future_row = int(first["row_index"] + 1)
    if pd.notna(tampered.loc[future_row, "hold_label"]):
        tampered.loc[future_row, "hold_label"] = 1 - int(tampered.loc[future_row, "hold_label"])
    replay = chronological_long_hold_exit_panel(
        frame, tampered, ["momentum", "rv"], target_spec, oos_spec
    )

    assert replay.iloc[0]["row_index"] == first["row_index"]
    assert replay.iloc[0]["learned_hold_probability"] == pytest.approx(first["learned_hold_probability"])
    assert replay.iloc[0]["learned_action"] == first["learned_action"]


def test_task_specific_baselines_share_identical_realized_outcomes():
    frame = _frame()
    # Force a peak then decision-bar retracement on a row that will be OOS eligible.
    frame.loc[8:10, "close"] = [103.0, 108.0, 104.0]
    frame.loc[8, "atr_points"] = 3.0
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2, cost_points=0.5)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    oos_spec = HoldExitOOSSpec(min_train_rows=3, trailing_points=2.0, atr_multiple=2.0)

    panel = chronological_long_hold_exit_panel(
        frame, targets, ["momentum", "rv"], target_spec, oos_spec
    )
    row = panel.loc[panel["row_index"] == 8].iloc[0]
    assert row["trailing_action"] == "EXIT"
    assert row["atr_trailing_action"] == "HOLD"
    assert row["trailing_realized_points"] == pytest.approx(row["pnl_if_exit_points"])
    assert row["atr_trailing_realized_points"] == pytest.approx(row["pnl_if_hold_points"])

    summary = summarize_hold_exit_panel(panel)
    assert set(summary["policy"]) == {"learned", "fixed_horizon", "trailing", "atr_trailing"}
    assert (summary["rows"] == len(panel)).all()


def test_future_target_columns_remain_forbidden_as_features():
    frame = _frame()
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    frame["hold_incremental_points"] = targets["hold_incremental_points"]
    with pytest.raises(ValueError, match="future/evaluation"):
        chronological_long_hold_exit_panel(
            frame,
            targets,
            ["momentum", "hold_incremental_points"],
            target_spec,
            HoldExitOOSSpec(min_train_rows=3),
        )
