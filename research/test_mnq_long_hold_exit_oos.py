import numpy as np
import pandas as pd
import pytest

import research.mnq_long_hold_exit_oos as oos
from research.mnq_long_hold_exit_oos import (
    HoldExitOOSSpec,
    chronological_long_hold_exit_panel,
    summarize_hold_exit_panel,
)
from research.mnq_long_hold_exit_targets import LongHoldExitSpec, materialize_long_hold_exit_targets


def _frame(n: int = 32) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="12min", tz="UTC"),
        "close": 100.0 + 0.15 * x + np.sin(x / 2.0),
        "momentum": np.sin(x / 3.0),
        "rv": 1.0 + 0.05 * np.cos(x / 4.0),
        "atr_points": 1.5 + 0.1 * np.sin(x / 5.0),
    })


def test_oos_training_is_purged_to_labels_known_by_decision_time():
    frame = _frame()
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2, cost_points=0.5)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    panel = chronological_long_hold_exit_panel(frame, targets, ["momentum", "rv"], target_spec, HoldExitOOSSpec(min_train_rows=3, trailing_points=1.0, atr_multiple=2.0))
    assert not panel.empty
    assert (panel["train_last_row"] + target_spec.horizon_bars <= panel["decision_row"]).all()
    assert (panel["train_last_decision_row"] == panel["train_last_row"] + target_spec.decision_bars).all()


def test_future_label_mutation_cannot_change_earliest_oos_prediction():
    frame = _frame()
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2, cost_points=0.5)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    spec = HoldExitOOSSpec(min_train_rows=3)
    original = chronological_long_hold_exit_panel(frame, targets, ["momentum", "rv"], target_spec, spec)
    first = original.iloc[0]
    tampered = targets.copy()
    future_row = int(first["row_index"] + 1)
    if pd.notna(tampered.loc[future_row, "hold_label"]):
        tampered.loc[future_row, "hold_label"] = 1 - int(tampered.loc[future_row, "hold_label"])
    replay = chronological_long_hold_exit_panel(frame, tampered, ["momentum", "rv"], target_spec, spec)
    assert replay.iloc[0]["learned_hold_probability"] == pytest.approx(first["learned_hold_probability"])
    assert replay.iloc[0]["learned_action"] == first["learned_action"]


def test_learned_prediction_uses_decision_state_not_entry_state(monkeypatch):
    """Prove row wiring directly instead of inferring it through a clipped probability.

    The old regression mutated features and expected the first predicted probability to
    change. That can false-fail when the real ridge prediction is saturated at the 0/1
    clip boundary. Instrumenting the predictor proves which causal feature row is passed
    without changing the production implementation.
    """
    frame = _frame(40)
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2, cost_points=0.5)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    spec = HoldExitOOSSpec(min_train_rows=3)
    seen_rows: list[np.ndarray] = []

    def capture_predict(row, fit):
        seen_rows.append(np.asarray(row, dtype=float).copy())
        return 0.5

    monkeypatch.setattr(oos, "_predict_probability", capture_predict)
    base = chronological_long_hold_exit_panel(frame, targets, ["momentum", "rv"], target_spec, spec)
    first = base.iloc[0]
    entry_i = int(first["row_index"])
    decision_i = int(first["decision_row"])
    expected_decision = frame.loc[decision_i, ["momentum", "rv"]].to_numpy(float)
    assert np.allclose(seen_rows[0], expected_decision)

    entry_mutated = frame.copy()
    entry_mutated.loc[entry_i, ["momentum", "rv"]] = [999.0, 999.0]
    seen_rows.clear()
    chronological_long_hold_exit_panel(entry_mutated, targets, ["momentum", "rv"], target_spec, spec)
    assert np.allclose(seen_rows[0], expected_decision)

    decision_mutated = frame.copy()
    decision_mutated.loc[decision_i, ["momentum", "rv"]] = [999.0, 999.0]
    seen_rows.clear()
    chronological_long_hold_exit_panel(decision_mutated, targets, ["momentum", "rv"], target_spec, spec)
    assert np.allclose(seen_rows[0], np.array([999.0, 999.0]))


def test_atr_trailing_uses_decision_atr_not_entry_atr():
    frame = _frame()
    frame.loc[8:10, "close"] = [103.0, 108.0, 104.0]
    frame.loc[8, "atr_points"] = 100.0
    frame.loc[10, "atr_points"] = 1.0
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2, cost_points=0.5)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    panel = chronological_long_hold_exit_panel(frame, targets, ["momentum", "rv"], target_spec, HoldExitOOSSpec(min_train_rows=3, trailing_points=2.0, atr_multiple=2.0))
    row = panel.loc[panel["row_index"] == 8].iloc[0]
    assert row["trailing_action"] == "EXIT"
    assert row["atr_trailing_action"] == "EXIT"
    assert row["atr_trailing_realized_points"] == pytest.approx(row["pnl_if_exit_points"])
    summary = summarize_hold_exit_panel(panel)
    assert set(summary["policy"]) == {"learned", "fixed_horizon", "trailing", "atr_trailing"}


def test_future_target_columns_remain_forbidden_as_features():
    frame = _frame()
    target_spec = LongHoldExitSpec(horizon_bars=6, decision_bars=2)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    frame["hold_incremental_points"] = targets["hold_incremental_points"]
    with pytest.raises(ValueError, match="future/evaluation"):
        chronological_long_hold_exit_panel(frame, targets, ["momentum", "hold_incremental_points"], target_spec, HoldExitOOSSpec(min_train_rows=3))
