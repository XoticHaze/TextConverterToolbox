from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

DEEP_SOURCE_TIMEZONE = "America/New_York"
DEEP_SOURCE_BAR_INTERVAL = pd.Timedelta(minutes=1)
DEEP_SOURCE_BAR_LABEL = "right_close"
CONTRACT_RE = re.compile(r"^MNQ (?P<month>\d{2})-(?P<year>\d{2})$")


def normalize_deep_bar_timestamps(values: pd.Series) -> pd.Series:
    naive = pd.to_datetime(values, errors="raise")
    if getattr(naive.dt, "tz", None) is not None:
        raise RuntimeError("deep MNQ timestamp unexpectedly already timezone-aware")
    localized = naive.dt.tz_localize(DEEP_SOURCE_TIMEZONE, ambiguous="infer", nonexistent="raise")
    return localized.dt.tz_convert("UTC") - DEEP_SOURCE_BAR_INTERVAL


def load_deep(root: Path) -> pd.DataFrame:
    nodes=[]; files=sorted(root.glob("MNQ */*.Last.csv"))
    if not files: raise RuntimeError(f"no Last.csv files under {root}")
    expected=["datetime","open","high","low","close","volume"]
    for path in files:
        contract=path.parent.name
        if not CONTRACT_RE.fullmatch(contract): continue
        frame=pd.read_csv(path)
        if list(frame.columns)!=expected: raise RuntimeError(f"{path}: unexpected schema {list(frame.columns)}")
        if frame.empty: continue
        frame["timestamp"]=normalize_deep_bar_timestamps(frame["datetime"])
        for c in ["open","high","low","close","volume"]: frame[c]=pd.to_numeric(frame[c],errors="raise")
        if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
            raise RuntimeError(f"{path}: invalid normalized UTC timestamp order")
        frame["symbol"]=contract
        nodes.append(frame[["timestamp","open","high","low","close","volume","symbol"]])
    if not nodes: raise RuntimeError("no usable deep MNQ rows")
    return pd.concat(nodes,ignore_index=True).sort_values(["timestamp","symbol"]).reset_index(drop=True)


def contract_receipt() -> dict[str,str]:
    return {
        "source":"mbytes21/MNQ_DATA Last.csv",
        "source_timestamp_timezone":DEEP_SOURCE_TIMEZONE,
        "source_bar_label":DEEP_SOURCE_BAR_LABEL,
        "normalized_timestamp":"UTC bar start",
        "normalization":"tz_localize(America/New_York,DST-aware)->UTC - 1 minute",
        "evidence_winter":"2026-01-21/22 OHLCV parity vs AXB UTC",
        "evidence_summer":"2025-06-10 OHLCV parity vs licensed MNQ UTC",
    }
