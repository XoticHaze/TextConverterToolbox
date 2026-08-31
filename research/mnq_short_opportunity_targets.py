from __future__ import annotations

"""Causal evaluation targets for an offline MNQ short-opportunity specialist.

Research only. These future outcomes are labels/evaluation evidence and are never
short-execution, broker, StrategySpec, or promotion authority.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


FUTURE_ONLY_COLUMNS = frozenset({
    "short_label",
    "short_forward_points",
    "short_mfe_points",
    "short_mae_points",
    "target_resolution_row",
})


@dataclass(frozen=True)
class ShortOpportunitySpec:
    horizon_bars: int = 24
    cost_points: float = 1.0
    min_edge_points: float = 0.0
    adverse_limit_points: float | None = None

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be positive")
        if self.cost_points < 0 or self.min_edge_points < 0:
            raise ValueError("cost/edge must be non-negative")
        if self.adverse_limit_points is not None and self.adverse_limit_points <= 0:
            raise ValueError("adverse_limit_points must be positive when set")


def assert_short_feature_fence(feature_columns: list[str]) -> None:
    leaked = sorted(set(feature_columns) & FUTURE_ONLY_COLUMNS)
    if leaked:
        raise ValueError(f"future short outcomes are not feature-eligible: {leaked}")


def materialize_short_opportunity_targets(frame: pd.DataFrame, spec: ShortOpportunitySpec) -> pd.DataFrame:
    """Materialize downside-specific opportunity outcomes for every causal entry row.

    Short P&L is entry minus future price. MFE is the best downside excursion and
    MAE is the worst upside excursion during the horizon. A positive label requires
    terminal downside edge to clear cost + minimum edge and, when configured, an
    adverse-excursion limit. This deliberately does not obtain short labels by
    negating a long classifier.
    """
    required = {"high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")

    values = frame[["high", "low", "close"]].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("OHLC inputs contain non-finite values")

    high = values["high"].to_numpy()
    low = values["low"].to_numpy()
    close = values["close"].to_numpy()
    n = len(frame)
    out = pd.DataFrame(index=frame.index, columns=[
        "short_label", "short_forward_points", "short_mfe_points",
        "short_mae_points", "target_resolution_row",
    ], dtype=float)

    hurdle = spec.cost_points + spec.min_edge_points
    for i in range(n):
        end = i + spec.horizon_bars
        if end >= n:
            continue
        entry = close[i]
        future_high = float(np.max(high[i + 1 : end + 1]))
        future_low = float(np.min(low[i + 1 : end + 1]))
        forward = float(entry - close[end])
        mfe = float(entry - future_low)
        mae = float(max(0.0, future_high - entry))
        passes_adverse = spec.adverse_limit_points is None or mae <= spec.adverse_limit_points
        out.loc[frame.index[i]] = [
            int(forward >= hurdle and passes_adverse),
            forward,
            mfe,
            mae,
            end,
        ]

    return out
