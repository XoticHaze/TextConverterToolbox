"""Synthetic original-index to compressed-index causal clock admission, no licensed MNQ."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.run_mnq_short_opportunity_deep import align_short_target_clock


def _fixture(n: int = 90, prefix: int = 7, horizon: int = 24):
    row = np.arange(n, dtype=float)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC"),
        "f1": row + 1,
        "f2": np.sin(row / 7),
    })
    frame.loc[: prefix - 1, ["f1", "f2"]] = np.nan
    targets = pd.DataFrame({
        "short_label": (row % 2).astype(float),
        "short_forward_points": row / 10,
        "target_resolution_row": row + horizon,
    })
    targets.loc[n - horizon :, :] = np.nan
    return frame, targets


def test_leading_warmup_rebases_original_resolution_clock_exactly():
    frame, targets = _fixture()
    aligned = align_short_target_clock(frame, targets, ["f1", "f2"])
    assert len(aligned) == len(frame) - 7
    assert aligned.loc[0, "timestamp"] == frame.loc[7, "timestamp"]
    assert aligned.loc[0, "target_resolution_row"] == 24
    for decision, expected in [(31, 7), (50, 26)]:
        eligible = aligned["target_resolution_row"].notna() & (aligned["target_resolution_row"] < decision)
        assert int(eligible.sum()) == expected
    assert aligned["short_forward_points"].iloc[0] == targets["short_forward_points"].iloc[7]


def test_interior_feature_gap_rejected_without_label_compression():
    frame, targets = _fixture()
    frame.loc[31, "f2"] = np.nan
    with pytest.raises(ValueError, match="interior feature gaps"):
        align_short_target_clock(frame, targets, ["f1", "f2"])


def test_zero_prefix_preserves_resolution_and_rejects_fractional_clock():
    frame, targets = _fixture(prefix=0)
    aligned = align_short_target_clock(frame, targets, ["f1", "f2"])
    assert aligned["target_resolution_row"].equals(targets["target_resolution_row"])
    targets.loc[3, "target_resolution_row"] = 3.5
    with pytest.raises(ValueError, match="finite integer"):
        align_short_target_clock(frame, targets, ["f1", "f2"])
