from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.deep_mnq_calendar_coverage_audit import coverage, largest_gaps, quarter_label
from research.deep_mnq_source_contract import CONTRACT_RE, contract_receipt, load_deep
from research.licensed_mnq_expanded_validation import _contract_key, _load_aggregate, _roll_schedule, _stitch
from research.mnq_external_transfer_validation import deep_roll_schedule, stitch_deep

DEEP_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
LICENSED_DATASET = "brandenmorris/mnq-1m-q4-2022-q4-2025-ts-ohlcv-sym"
LICENSED_VERSION = 1
LICENSED_LICENSE = "CC BY 4.0"
LICENSED_ZIP_SHA256 = "633eb4338a3aa60aedb542b12085acca29ff237c7cadff65442a638466f37667"
MIN_SAME_CONTRACT_OVERLAP = 10_000
MAX_MEDIAN_CLOSE_DELTA = 0.25
MAX_P99_CLOSE_DELTA = 1.0
MAX_POST_UNION_GAP_MINUTES = 14 * 24 * 60
MIN_Q4_12MIN_BARS = 5_000


def _canonical_deep_contract(symbol: str) -> str:
    m = CONTRACT_RE.fullmatch(str(symbol))
    if not m:
        raise RuntimeError(f"unexpected deep contract {symbol!r}")
    month = int(m.group("month"))
    year = 2000 + int(m.group("year"))
    return f"MNQ-{year:04d}-{month:02d}"


def _canonical_licensed_contract(symbol: str) -> str:
    year, month = _contract_key(str(symbol))
    return f"MNQ-{year:04d}-{month:02d}"


def _decorate(frame: pd.DataFrame, source_dataset: str, canonicalizer) -> pd.DataFrame:
    out = frame[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    if out["timestamp"].duplicated().any():
        dup = out.loc[out["timestamp"].duplicated(keep=False), "timestamp"].head(10).astype(str).tolist()
        raise RuntimeError(f"{source_dataset}: stitched duplicate timestamps {dup}")
    if not out["timestamp"].is_monotonic_increasing:
        out = out.sort_values("timestamp").reset_index(drop=True)
    out["source_dataset"] = source_dataset
    out["vendor_contract"] = out["symbol"].astype(str)
    out["source_contract"] = out["symbol"].map(canonicalizer)
    return out.drop(columns=["symbol"])


def build_sources(deep_root: Path, licensed_aggregate: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    deep_raw = load_deep(deep_root)
    deep_schedule = deep_roll_schedule(deep_raw)
    deep_stitched = stitch_deep(deep_raw, deep_schedule)
    deep = _decorate(deep_stitched, "deep_primary", _canonical_deep_contract)

    licensed_raw = _load_aggregate(licensed_aggregate)
    licensed_schedule = _roll_schedule(licensed_raw)
    licensed_stitched = _stitch(licensed_raw, licensed_schedule)
    licensed = _decorate(licensed_stitched, "licensed_fill", _canonical_licensed_contract)

    source_receipt = {
        "deep": {
            "source": f"mbytes21/MNQ_DATA@{DEEP_COMMIT}",
            "timestamp_contract": contract_receipt(),
            "raw_rows": int(len(deep_raw)),
            "stitched_rows": int(len(deep)),
            "roll_sessions": int(len(deep_schedule)),
        },
        "licensed": {
            "source": LICENSED_DATASET,
            "version": LICENSED_VERSION,
            "license": LICENSED_LICENSE,
            "zip_sha256": LICENSED_ZIP_SHA256,
            "raw_rows": int(len(licensed_raw)),
            "stitched_rows": int(len(licensed)),
            "roll_sessions": int(len(licensed_schedule)),
        },
    }
    return deep, licensed, source_receipt


def overlap_audit(deep: pd.DataFrame, licensed: pd.DataFrame) -> dict:
    joined = deep.merge(licensed, on="timestamp", suffixes=("_deep", "_licensed"), how="inner")
    if joined.empty:
        raise RuntimeError("deep/licensed sources have no timestamp overlap")
    same = joined[joined["source_contract_deep"] == joined["source_contract_licensed"]].copy()
    if len(same) < MIN_SAME_CONTRACT_OVERLAP:
        raise RuntimeError(f"insufficient same-contract overlap: {len(same)}")

    metrics: dict[str, dict] = {}
    for col in ("open", "high", "low", "close", "volume"):
        delta = (same[f"{col}_deep"].astype(float) - same[f"{col}_licensed"].astype(float)).abs()
        metrics[col] = {
            "median_abs_delta": float(delta.median()),
            "p95_abs_delta": float(delta.quantile(0.95)),
            "p99_abs_delta": float(delta.quantile(0.99)),
            "max_abs_delta": float(delta.max()),
            "exact_fraction": float((delta == 0).mean()),
        }

    close = metrics["close"]
    gate = {
        "same_contract_overlap_minutes": len(same) >= MIN_SAME_CONTRACT_OVERLAP,
        "median_close_delta": close["median_abs_delta"] <= MAX_MEDIAN_CLOSE_DELTA,
        "p99_close_delta": close["p99_abs_delta"] <= MAX_P99_CLOSE_DELTA,
    }
    return {
        "overlap_minutes": int(len(joined)),
        "same_contract_overlap_minutes": int(len(same)),
        "same_contract_fraction": float(len(same) / len(joined)),
        "different_contract_overlap_minutes": int(len(joined) - len(same)),
        "ohlcv_absolute_delta": metrics,
        "gate": gate,
        "pass": bool(all(gate.values())),
    }


def build_union(deep: pd.DataFrame, licensed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deep_timestamps = pd.Index(deep["timestamp"])
    fill = licensed.loc[~licensed["timestamp"].isin(deep_timestamps)].copy()
    union = pd.concat([deep, fill], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if union["timestamp"].duplicated().any():
        raise RuntimeError("union contains duplicate timestamps")
    if not union["timestamp"].is_monotonic_increasing:
        raise RuntimeError("union timestamps are not monotonic")
    return union, fill


def union_12min_bars(union: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    w = union.set_index("timestamp")
    bars = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        observed_minutes=("close", "count"),
        source_dataset=("source_dataset", "first"),
        source_dataset_last=("source_dataset", "last"),
        source_contract=("source_contract", "first"),
        source_contract_last=("source_contract", "last"),
    )
    bars = bars[bars["observed_minutes"] > 0].copy()
    mixed = (bars["source_dataset"] != bars["source_dataset_last"]) | (
        bars["source_contract"] != bars["source_contract_last"]
    )
    dropped = int(mixed.sum())
    bars = bars.loc[~mixed].drop(columns=["source_dataset_last", "source_contract_last"]).reset_index()
    return bars, dropped


def _quarter_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    q = quarter_label(frame["timestamp"])
    return {str(k): int(v) for k, v in q.value_counts().sort_index().items()}


def _seam_audit(union: pd.DataFrame) -> dict:
    if len(union) < 2:
        return {"source_transitions": 0, "rows": []}
    changed = union["source_dataset"].ne(union["source_dataset"].shift(1))
    idxs = np.flatnonzero(changed.to_numpy())
    idxs = idxs[idxs > 0]
    rows = []
    for i in idxs:
        prev = union.iloc[int(i - 1)]
        cur = union.iloc[int(i)]
        rows.append({
            "previous_timestamp": pd.Timestamp(prev["timestamp"]).isoformat(),
            "timestamp": pd.Timestamp(cur["timestamp"]).isoformat(),
            "previous_source": str(prev["source_dataset"]),
            "source": str(cur["source_dataset"]),
            "previous_contract": str(prev["source_contract"]),
            "contract": str(cur["source_contract"]),
            "gap_minutes": float((cur["timestamp"] - prev["timestamp"]) / pd.Timedelta(minutes=1)),
            "close_change_points": float(cur["close"] - prev["close"]),
            "close_return": float(cur["close"] / prev["close"] - 1.0) if float(prev["close"]) != 0 else None,
        })
    return {"source_transitions": int(len(rows)), "rows": rows}


def _union_fingerprint(union: pd.DataFrame) -> str:
    cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "source_dataset", "vendor_contract", "source_contract",
    ]
    digest = hashlib.sha256()
    for start in range(0, len(union), 100_000):
        chunk = union.iloc[start : start + 100_000][cols].copy()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        text = chunk.to_csv(index=False, header=(start == 0), float_format="%.10g", lineterminator="\n")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _post_union_gap_gate(union: pd.DataFrame) -> tuple[dict, list[dict]]:
    window = union[
        (union["timestamp"] >= pd.Timestamp("2022-01-01", tz="UTC"))
        & (union["timestamp"] < pd.Timestamp("2026-01-01", tz="UTC"))
    ].copy()
    gaps = largest_gaps(window, limit=40)
    material = [g for g in gaps if g["gap_minutes"] > MAX_POST_UNION_GAP_MINUTES]
    max_gap = max((float(g["gap_minutes"]) for g in gaps), default=0.0)
    return {
        "max_gap_minutes": max_gap,
        "maximum_allowed_gap_minutes": MAX_POST_UNION_GAP_MINUTES,
        "material_gap_count": int(len(material)),
        "pass": len(material) == 0,
    }, gaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--licensed-aggregate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    deep, licensed, sources = build_sources(args.deep_root, args.licensed_aggregate)
    overlap = overlap_audit(deep, licensed)
    if not overlap["pass"]:
        raise RuntimeError("licensed fill source failed frozen overlap compatibility gate")

    union, fill = build_union(deep, licensed)
    bars12, dropped_mixed_bars = union_12min_bars(union)
    gap_gate, union_gaps = _post_union_gap_gate(union)
    bars_cov = coverage(bars12)
    q4_gate = {
        str(year): int(bars_cov.get(f"{year}Q4", {}).get("rows", 0)) >= MIN_Q4_12MIN_BARS
        for year in (2023, 2024, 2025)
    }
    coverage_gate = {
        "post_union_gap": bool(gap_gate["pass"]),
        "q4_12min_support_2023_2025": bool(all(q4_gate.values())),
    }
    accepted = bool(overlap["pass"] and all(coverage_gate.values()))

    result = {
        "schema": "foundry.deep_mnq_licensed_gap_fill.v1",
        "research_only": True,
        "promotion_authority": False,
        "authority_issue": "XoticHaze/research-foundry#78",
        "sources": sources,
        "union_policy": "deep primary by normalized UTC minute; licensed contributes only missing timestamps; no interpolation/back-adjustment/price modification",
        "overlap_audit": overlap,
        "primary_rows": int(len(deep)),
        "licensed_selected_rows": int(len(licensed)),
        "filled_minutes": int(len(fill)),
        "filled_minutes_by_quarter": _quarter_counts(fill),
        "union_rows": int(len(union)),
        "union_rows_by_quarter": _quarter_counts(union),
        "source_rows": {str(k): int(v) for k, v in union["source_dataset"].value_counts().sort_index().items()},
        "source_contract_count": int(union["source_contract"].nunique()),
        "seam_audit": _seam_audit(union),
        "post_union_gap_gate": gap_gate,
        "post_union_largest_gaps": union_gaps[:20],
        "bars_12min": {
            "rows": int(len(bars12)),
            "dropped_mixed_source_or_contract_bars": dropped_mixed_bars,
            "quarter_coverage": bars_cov,
            "q4_minimum_rows": MIN_Q4_12MIN_BARS,
            "q4_gate": q4_gate,
        },
        "coverage_gate": coverage_gate,
        "union_fingerprint_sha256": _union_fingerprint(union),
        "accepted_for_full_calendar_revalidation": accepted,
        "next_action": (
            "rerun earned MNQ hypotheses against this union receipt before any full-calendar claim"
            if accepted
            else "inspect already-owned MM-IBKR/IBKR history for remaining gaps before acquiring another source"
        ),
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("DEEP_MNQ_LICENSED_GAP_FILL=" + ("PASS" if accepted else "FAIL"))
    print("FILLED_MINUTES=" + str(len(fill)))
    print("FILLED_BY_QUARTER=" + json.dumps(result["filled_minutes_by_quarter"], sort_keys=True))
    print("OVERLAP=" + json.dumps(overlap, sort_keys=True))
    print("POST_UNION_GAP_GATE=" + json.dumps(gap_gate, sort_keys=True))
    print("Q4_GATE=" + json.dumps(q4_gate, sort_keys=True))
    print("UNION_FINGERPRINT_SHA256=" + result["union_fingerprint_sha256"])
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
