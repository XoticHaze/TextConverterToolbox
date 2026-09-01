from __future__ import annotations

"""Causal target materialization for the MNQ long HOLD/EXIT specialist.

Research-only. This module defines outcomes available only for training/evaluation;
it does not define runtime exit authority or permit target columns as features.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongHoldExitSpec:
    horizon_bars: int = 24
    decision_bars: int = 6
    cost_points: float = 1.0
    min_remaining_edge_points: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_bars <= 1:
            raise ValueError("horizon_bars must exceed 1")
        if not 0 < self.decision_bars < self.horizon_bars:
            raise ValueError("decision_bars must be inside the position horizon")
        if self.cost_points < 0:
            raise ValueError("cost_points must be non-negative")


def materialize_long_hold_exit_targets(frame: pd.DataFrame, spec: LongHoldExitSpec) -> pd.DataFrame:
    """Return evaluation-only long-position trajectory targets.

    Every row is treated as a hypothetical long entry. At ``decision_bars`` we
    compare exiting immediately with holding to ``horizon_bars``. The label is
    therefore conditional on already being long, not an entry or short label.

    ``hold_incremental_points`` is the future-only incremental P&L from the
    decision timestamp to horizon. HOLD must clear the configured remaining
    edge plus the incremental exit-cost sensitivity. Deterministic risk exits
    remain outside this learned target and retain veto authority downstream.
    """
    if "close" not in frame:
        raise ValueError("frame must contain close")
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    if not np.isfinite(close.to_numpy()).all():
        raise ValueError("close contains non-finite values")

    decision = close.shift(-spec.decision_bars)
    horizon = close.shift(-spec.horizon_bars)
    entry = close

    out = pd.DataFrame(index=frame.index)
    if "timestamp" in frame:
        out["timestamp"] = frame["timestamp"]
    out["entry_price"] = entry
    out["decision_price"] = decision
    out["horizon_price"] = horizon
    out["pnl_if_exit_points"] = decision - entry - spec.cost_points
    out["pnl_if_hold_points"] = horizon - entry - spec.cost_points
    out["hold_incremental_points"] = horizon - decision
    out["remaining_edge_after_cost_points"] = out["hold_incremental_points"] - spec.cost_points

    resolved = decision.notna() & horizon.notna()
    threshold = spec.min_remaining_edge_points + spec.cost_points
    label = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    label.loc[resolved] = (out.loc[resolved, "hold_incremental_points"] > threshold).astype("int8")
    out["hold_label"] = label
    out["target_resolution_bars"] = spec.horizon_bars
    out["decision_bars"] = spec.decision_bars
    out["research_only"] = True
    return out


def assert_feature_fence(feature_columns: list[str]) -> None:
    forbidden = {
        "decision_price", "horizon_price", "pnl_if_exit_points", "pnl_if_hold_points",
        "hold_incremental_points", "remaining_edge_after_cost_points", "hold_label",
    }
    leaked = sorted(forbidden.intersection(feature_columns))
    if leaked:
        raise ValueError(f"future/evaluation columns are not feature-eligible: {leaked}")
