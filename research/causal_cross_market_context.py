"""Causal cross-market context features for specialist research.

Research-only. This module deliberately uses backward as-of joins so a target
instrument at time t can only consume context observations timestamped <= t.
It is intended for ablations such as NQ/MNQ conditioned on ES, RTY, GC, CL or
other independently materialized roots without creating a runtime authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextSpec:
    return_bars: tuple[int, ...] = (1, 5, 20)
    vol_bars: int = 20
    max_staleness: pd.Timedelta = pd.Timedelta("15min")


def _normalize(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"timestamp", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    out = frame[["timestamp", "close"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    out["close"] = pd.to_numeric(out["close"], errors="raise")
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if not out["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{name}: timestamps are not monotonic")
    if (out["close"] <= 0).any():
        raise ValueError(f"{name}: close must be positive")
    return out.reset_index(drop=True)


def build_context_features(
    target: pd.DataFrame,
    contexts: Mapping[str, pd.DataFrame],
    spec: ContextSpec = ContextSpec(),
) -> pd.DataFrame:
    """Return target timestamps plus causal context features.

    Every context observation is joined with direction='backward'. A feature is
    invalidated when its source observation is older than max_staleness. The
    returned ``<root>__age_seconds`` column makes coverage/staleness auditable.
    Returns and realized volatility are computed within each context series
    before joining, never from target/future timestamps.
    """
    base = _normalize(target, "target")[["timestamp"]]
    for root, raw in contexts.items():
        if not root or "__" in root:
            raise ValueError(f"invalid context root {root!r}")
        ctx = _normalize(raw, root)
        source_ts = f"{root}__source_timestamp"
        ctx[source_ts] = ctx["timestamp"]
        logp = np.log(ctx["close"])
        feature_cols: list[str] = []
        for bars in spec.return_bars:
            if bars <= 0:
                raise ValueError("return_bars must be positive")
            col = f"{root}__logret_{bars}"
            ctx[col] = logp.diff(bars)
            feature_cols.append(col)
        vol_col = f"{root}__rv_{spec.vol_bars}"
        if spec.vol_bars < 2:
            raise ValueError("vol_bars must be >= 2")
        ctx[vol_col] = logp.diff().rolling(spec.vol_bars, min_periods=spec.vol_bars).std()
        feature_cols.append(vol_col)
        payload = ctx[["timestamp", source_ts, *feature_cols]]
        base = pd.merge_asof(base, payload, on="timestamp", direction="backward", allow_exact_matches=True)
        age_col = f"{root}__age_seconds"
        base[age_col] = (base["timestamp"] - base[source_ts]).dt.total_seconds()
        stale = base[age_col] > spec.max_staleness.total_seconds()
        base.loc[stale, [source_ts, *feature_cols]] = np.nan
    return base


def assert_causal_context(features: pd.DataFrame) -> None:
    """Fail closed if any joined source timestamp is later than target time."""
    target_ts = pd.to_datetime(features["timestamp"], utc=True, errors="raise")
    for col in (c for c in features.columns if c.endswith("__source_timestamp")):
        source = pd.to_datetime(features[col], utc=True, errors="coerce")
        bad = source.notna() & (source > target_ts)
        if bad.any():
            raise AssertionError(f"future context detected in {col}: {int(bad.sum())} rows")
