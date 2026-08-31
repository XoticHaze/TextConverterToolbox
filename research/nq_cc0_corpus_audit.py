from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ET_ZONE = "America/New_York"
NUMERIC = ["open", "high", "low", "close", "volume"]


def canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in frame.columns:
        key = "".join(ch for ch in c.lower() if ch.isalnum())
        if "timestamp" in key and "et" in key:
            mapping[c] = "timestamp_et"
        elif key in {"open", "high", "low", "close", "volume", "vwaprth", "vwapeth"}:
            mapping[c] = key.replace("vwaprth", "vwap_rth").replace("vwapeth", "vwap_eth")
    out = frame.rename(columns=mapping)
    needed = {"timestamp_et", *NUMERIC}
    missing = sorted(needed - set(out.columns))
    if missing:
        raise RuntimeError(f"missing required columns {missing}; observed={list(frame.columns)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-normalized", type=Path, required=True)
    args = ap.parse_args()

    raw_bytes = args.input.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    frame = canonical_columns(pd.read_csv(args.input))
    source_rows = int(len(frame))
    nulls = {c: int(frame[c].isna().sum()) for c in frame.columns}

    naive = pd.to_datetime(frame["timestamp_et"], errors="raise")
    naive_dupes = int(naive.duplicated(keep=False).sum())
    try:
        localized = pd.DatetimeIndex(naive).tz_localize(ET_ZONE, ambiguous="infer", nonexistent="raise")
    except Exception as exc:
        raise RuntimeError(f"ET->UTC localization failed; refusing to guess DST folds: {exc}") from exc
    utc = localized.tz_convert("UTC")

    out = pd.DataFrame({"timestamp": utc})
    for c in NUMERIC:
        out[c] = pd.to_numeric(frame[c], errors="raise")
    for c in ("vwap_rth", "vwap_eth"):
        if c in frame.columns:
            out[c] = pd.to_numeric(frame[c], errors="coerce")

    utc_dupes = int(out["timestamp"].duplicated(keep=False).sum())
    monotonic_before = bool(out["timestamp"].is_monotonic_increasing)
    out = out.sort_values("timestamp").reset_index(drop=True)
    if utc_dupes:
        raise RuntimeError(f"UTC conversion produced {utc_dupes} duplicated timestamp rows")

    dsec = out["timestamp"].diff().dt.total_seconds().dropna()
    gap_counts = {
        "one_minute": int((dsec == 60).sum()),
        "under_one_minute": int((dsec < 60).sum()),
        "over_one_minute": int((dsec > 60).sum()),
        "over_30_minutes": int((dsec > 1800).sum()),
        "max_gap_seconds": float(dsec.max()) if len(dsec) else None,
    }

    o, h, l, c = (out[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    ohlc_bad = int(np.sum((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)))
    nonpositive_price = int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)))
    negative_volume = int(np.sum(out["volume"].to_numpy(float) < 0))

    prev = out["close"].shift(1)
    jump = (out["close"] / prev - 1.0).abs()
    top_idx = jump.nlargest(20).index
    top_jumps = [
        {
            "timestamp": out.loc[i, "timestamp"].isoformat(),
            "abs_close_return": float(jump.loc[i]),
            "previous_close": float(prev.loc[i]),
            "close": float(out.loc[i, "close"]),
        }
        for i in top_idx if pd.notna(jump.loc[i])
    ]

    bars = out.set_index("timestamp").resample("12min", origin="start_day", label="left", closed="left").agg(
        observed_minutes=("close", "count")
    )
    observed = bars.loc[bars["observed_minutes"] > 0, "observed_minutes"]
    twelve = {
        "bars": int(len(observed)),
        "full_12_minute_bars": int((observed == 12).sum()),
        "partial_bars": int((observed < 12).sum()),
        "median_observed_minutes": float(observed.median()) if len(observed) else None,
        "min_observed_minutes": int(observed.min()) if len(observed) else None,
    }

    args.output_normalized.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_normalized, index=False, compression="gzip")
    normalized_sha256 = hashlib.sha256(args.output_normalized.read_bytes()).hexdigest()

    result = {
        "schema": "foundry.nq_cc0_corpus_audit.v1",
        "research_only": True,
        "promotion_authority": False,
        "license": "CC0: Public Domain",
        "source": "tgtanalytics/nq-futures-1min-bar-2022-2025",
        "source_file": args.input.name,
        "source_sha256": source_sha256,
        "rows": source_rows,
        "source_nulls": nulls,
        "naive_et_duplicate_rows": naive_dupes,
        "utc_duplicate_rows": utc_dupes,
        "source_order_monotonic_after_dst_conversion": monotonic_before,
        "start_utc": out["timestamp"].min().isoformat(),
        "end_utc": out["timestamp"].max().isoformat(),
        "gap_counts": gap_counts,
        "ohlc_bad_rows": ohlc_bad,
        "nonpositive_price_rows": nonpositive_price,
        "negative_volume_rows": negative_volume,
        "abs_close_returns_over_2pct": int((jump > 0.02).sum()),
        "top_abs_close_jumps": top_jumps,
        "twelve_minute_materialization": twelve,
        "normalized_file": args.output_normalized.name,
        "normalized_sha256": normalized_sha256,
        "timezone_contract": "source timestamps interpreted as America/New_York wall clock; DST folds must be inferable from chronological sequence; ambiguous localization fails closed rather than guessing",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_CC0_CORPUS_AUDIT=PASS")
    print(f"ROWS={source_rows}")
    print(f"RANGE={result['start_utc']}..{result['end_utc']}")
    print(f"NAIVE_ET_DUPLICATE_ROWS={naive_dupes}")
    print(f"UTC_DUPLICATE_ROWS={utc_dupes}")
    print(f"GAPS_OVER_1MIN={gap_counts['over_one_minute']}")
    print(f"ABS_CLOSE_RETURNS_OVER_2PCT={result['abs_close_returns_over_2pct']}")
    print("SOURCE_SHA256=" + source_sha256)
    print("NORMALIZED_SHA256=" + normalized_sha256)
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
