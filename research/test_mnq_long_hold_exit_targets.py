import pandas as pd
import pytest

from research.mnq_long_hold_exit_targets import (
    LongHoldExitSpec,
    assert_feature_fence,
    materialize_long_hold_exit_targets,
)


def test_hold_exit_is_conditional_remaining_edge_not_entry_direction():
    frame = pd.DataFrame({"close": [100, 101, 102, 103, 106, 108, 109]})
    spec = LongHoldExitSpec(horizon_bars=4, decision_bars=2, cost_points=1.0)
    out = materialize_long_hold_exit_targets(frame, spec)
    # row 0: decision 102, horizon 106, remaining move 4 > 1pt hurdle => HOLD
    assert out.loc[0, "hold_label"] == 1
    assert out.loc[0, "hold_incremental_points"] == 4
    # target resolves only when both future timestamps exist
    assert pd.isna(out.loc[3, "hold_label"])


def test_hold_requires_incremental_edge_to_clear_cost_hurdle():
    frame = pd.DataFrame({"close": [100, 101, 102, 102.5, 102.75, 103]})
    spec = LongHoldExitSpec(horizon_bars=3, decision_bars=1, cost_points=2.0)
    out = materialize_long_hold_exit_targets(frame, spec)
    assert out.loc[0, "hold_incremental_points"] == 1.5
    assert out.loc[0, "hold_label"] == 0


def test_future_outcomes_are_explicitly_feature_ineligible():
    with pytest.raises(ValueError, match="future/evaluation"):
        assert_feature_fence(["rsi", "hold_incremental_points"])
    assert_feature_fence(["rsi", "rv_120"])
