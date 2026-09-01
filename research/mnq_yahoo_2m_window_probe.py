from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SYMBOL = "MNQ=F"
REQUEST_START = pd.Timestamp("2026-07-06T00:00:00Z")
REQUEST_END_EXCLUSIVE = pd.Timestamp("2026-08-29T00:00:00Z")
EARLIEST_ACCEPTABLE_FIRST = pd.Timestamp("2026-07-07T00:00:00Z")
LATEST_ACCEPTABLE_LAST = pd.Timestamp("2026-08-28T20:00:00Z")
CHUNKS = (
    ("2026-07-06", "2026-07-20"),
    ("2026-07-20", "2026-08-03"),
    ("2026-08-03", "2026-08-17"),
    ("2026-08-17", "2026-08-29"),
)
MIN_DISTINCT_DATES = 35
MIN_MEDIAN_DAILY_12M_BARS = 80
MIN_FULLISH_12M_FRACTION = 0.95


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if SYMBOL in frame.columns.get_level_values(-1):
            frame = frame.xs(SYMBOL, axis=1, level=-1)
        elif SYMBOL in frame.columns.get_level_values(0):
            frame = frame.xs(SYMBOL, axis=1, level=0)
        else:
            raise RuntimeError(f"unexpected Yahoo MultiIndex columns {frame.columns.tolist()[:8]}")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Yahoo missing required columns {missing}; got {list(frame.columns)}")
    if getattr(frame.index, "tz", None) is None:
        raise RuntimeError("Yahoo intraday index unexpectedly timezone-naive")
    frame = frame[required].copy()
    frame["timestamp"] = pd.to_datetime(frame.index, utc=True, errors="raise")
    for c in required:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.reset_index(drop=True).sort_values("timestamp")
    return frame[["timestamp", *required]]


def _download_chunk(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    raw = yf.download(
        SYMBOL,
        start=start,
        end=end,
        interval="2m",
        auto_adjust=False,
        prepost=True,
        progress=False,
        threads=False,
    )
    frame = _normalize(raw)
    return frame, {
        "requested_start": start,
        "requested_end_exclusive": end,
        "rows": int(len(frame)),
        "first_timestamp": frame["timestamp"].min().isoformat() if len(frame) else None,
        "last_timestamp": frame["timestamp"].max().isoformat() if len(frame) else None,
    }


def _canonical_sha(frame: pd.DataFrame) -> str:
    x = frame.copy()
    x["timestamp"] = x["timestamp"].map(lambda v: v.isoformat())
    material = x.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode()
    return hashlib.sha256(material).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    frames = []
    chunks = []
    for start, end in CHUNKS:
        frame, receipt = _download_chunk(start, end)
        chunks.append(receipt)
        if len(frame):
            frames.append(frame)
        print("CHUNK=" + json.dumps(receipt, sort_keys=True))

    if not frames:
        combined = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        combined = pd.concat(frames, ignore_index=True)
        duplicate_rows = int(combined.duplicated("timestamp", keep=False).sum())
        combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if not combined["timestamp"].is_monotonic_increasing:
            raise RuntimeError("combined Yahoo timestamps are not monotonic")
    duplicate_rows = int(locals().get("duplicate_rows", 0))

    if len(combined):
        w = combined.set_index("timestamp")
        bars = w.resample("12min", origin="start_day", label="left", closed="left").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            observed_2m=("close", "count"),
        )
        bars = bars[bars["observed_2m"] > 0].reset_index()
        date_counts = bars.groupby(bars["timestamp"].dt.date).size().astype(float)
        distinct_dates = int(len(date_counts))
        median_daily = float(date_counts.median()) if distinct_dates else 0.0
        fullish_fraction = float(np.mean(bars["observed_2m"].to_numpy(float) >= 5.0)) if len(bars) else 0.0
        first = combined["timestamp"].min()
        last = combined["timestamp"].max()
    else:
        bars = pd.DataFrame()
        distinct_dates = 0
        median_daily = 0.0
        fullish_fraction = 0.0
        first = None
        last = None

    checks = {
        "has_rows": bool(len(combined)),
        "first_timestamp_covers_requested_start": bool(first is not None and first <= EARLIEST_ACCEPTABLE_FIRST),
        "last_timestamp_covers_requested_end": bool(last is not None and last >= LATEST_ACCEPTABLE_LAST),
        "distinct_dates_ge_35": distinct_dates >= MIN_DISTINCT_DATES,
        "median_daily_12m_bars_ge_80": median_daily >= MIN_MEDIAN_DAILY_12M_BARS,
        "fullish_12m_fraction_ge_0p95": fullish_fraction >= MIN_FULLISH_12M_FRACTION,
    }
    result = {
        "schema": "foundry.mnq_yahoo_2m_window_probe.v1",
        "research_only": True,
        "promotion_authority": False,
        "model_evaluation_performed": False,
        "provider": "Yahoo Finance via yfinance",
        "provider_symbol": SYMBOL,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "request_window": {
            "start": REQUEST_START.isoformat(),
            "end_exclusive": REQUEST_END_EXCLUSIVE.isoformat(),
            "interval": "2m",
            "prepost": True,
            "chunks": chunks,
        },
        "combined": {
            "rows": int(len(combined)),
            "first_timestamp": first.isoformat() if first is not None else None,
            "last_timestamp": last.isoformat() if last is not None else None,
            "duplicate_timestamp_rows_before_dedupe": duplicate_rows,
            "canonical_csv_sha256": _canonical_sha(combined) if len(combined) else None,
            "bars_12m": int(len(bars)),
            "distinct_dates_with_12m_bars": distinct_dates,
            "median_daily_12m_bars": median_daily,
            "fraction_12m_bars_with_at_least_5_of_6_2m_samples": fullish_fraction,
        },
        "acceptance_checks": checks,
        "full_requested_window_available": bool(all(checks.values())),
        "raw_data_redistributed": False,
        "contract": "source-availability probe only; no features, targets, model fit, prediction, PnL, threshold selection, or promotion inference",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("FULL_REQUESTED_WINDOW_AVAILABLE=" + str(result["full_requested_window_available"]).lower())
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
