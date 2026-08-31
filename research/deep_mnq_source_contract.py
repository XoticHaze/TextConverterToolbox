from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DEEP_SOURCE_TIMEZONE = "America/New_York"
DEEP_SOURCE_BAR_INTERVAL = pd.Timedelta(minutes=1)
DEEP_SOURCE_BAR_LABEL = "right_close"
CONTRACT_RE = re.compile(r"^MNQ (?P<month>\d{2})-(?P<year>\d{2})$")


def normalize_deep_bar_timestamps(values: pd.Series) -> pd.Series:
    """Normalize deep Last.csv local close labels to UTC one-minute bar starts.

    Empirical contract is verified byte-for-byte against independent UTC sources:
    winter 2026 deep 00:01 == AXB 05:00 UTC and summer 2025 deep 00:01 ==
    licensed MNQ 04:00 UTC for consecutive OHLCV rows. Thus the source is
    America/New_York local bar-close labeling with DST, while comparators label
    the same minute by UTC bar start.
    """
    naive = pd.to_datetime(values, errors="raise")
    if getattr(naive.dt, "tz", None) is not None:
        raise RuntimeError("deep MNQ timestamp unexpectedly already timezone-aware")
    localized = naive.dt.tz_localize(
        DEEP_SOURCE_TIMEZONE,
        ambiguous="infer",
        nonexistent="raise",
    )
    return localized.dt.tz_convert("UTC") - DEEP_SOURCE_BAR_INTERVAL


def load_deep(root: Path) -> pd.DataFrame:
    nodes: list[pd.DataFrame] = []
    files = sorted(root.glob("MNQ */*.Last.csv"))
    if not files:
        raise RuntimeError(f"no Last.csv files under {root}")
    expected = ["datetime", "open", "high", "low", "close", "volume"]
    for path in files:
        contract = path.parent.name
        if not CONTRACT_RE.fullmatch(contract):
            continue
        frame = pd.read_csv(path)
        if list(frame.columns) != expected:
            raise RuntimeError(f"{path}: unexpected schema {list(frame.columns)}")
        if frame.empty:
            continue
        frame["timestamp"] = normalize_deep_bar_timestamps(frame["datetime"])
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame["timestamp"].duplicated().any():
            raise RuntimeError(f"{path}: duplicate UTC timestamps after normalization")
        if not frame["timestamp"].is_monotonic_increasing:
            raise RuntimeError(f"{path}: UTC timestamps not increasing after normalization")
        frame["symbol"] = contract
        nodes.append(frame[["timestamp", "open", "high", "low", "close", "volume", "symbol"]])
    if not nodes:
        raise RuntimeError("no usable deep MNQ rows")
    return pd.concat(nodes, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def contract_receipt() -> dict[str, str]:
    return {
        "source": "mbytes21/MNQ_DATA Last.csv",
        "source_timestamp_timezone": DEEP_SOURCE_TIMEZONE,
        "source_bar_label": DEEP_SOURCE_BAR_LABEL,
        "normalized_timestamp": "UTC bar start",
        "normalization": "tz_localize(America/New_York,DST-aware)->UTC - 1 minute",
        "evidence_winter": "2026-01-21/22 consecutive OHLCV parity vs axb0306/cme-futures-ohlc UTC",
        "evidence_summer": "2025-06-10 consecutive OHLCV parity vs licensed CC BY MNQ UTC",
    }
