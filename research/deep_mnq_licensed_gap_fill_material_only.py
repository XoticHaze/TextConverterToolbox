from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research.deep_mnq_licensed_gap_fill as base
from research.deep_mnq_calendar_coverage_audit import coverage

MATERIAL_GAP_MINUTES = 14 * 24 * 60
AMENDMENT = "research/deep_mnq_gap_fill_integrity_amendment_20260901.json"


def primary_material_gaps(deep: pd.DataFrame) -> list[dict]:
    ts = pd.to_datetime(deep["timestamp"], utc=True).sort_values().reset_index(drop=True)
    delta = ts.diff() / pd.Timedelta(minutes=1)
    gaps: list[dict] = []
    for idx in np.flatnonzero((delta > MATERIAL_GAP_MINUTES).fillna(False).to_numpy()):
        prior = ts.iloc[int(idx - 1)]
        nxt = ts.iloc[int(idx)]
        gaps.append(
            {
                "previous_primary_timestamp": prior,
                "next_primary_timestamp": nxt,
                "gap_minutes": float((nxt - prior) / pd.Timedelta(minutes=1)),
            }
        )
    return gaps


def build_material_gap_union(
    deep: pd.DataFrame, supplemental: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    gaps = primary_material_gaps(deep)
    pieces: list[pd.DataFrame] = []
    receipts: list[dict] = []
    for gap in gaps:
        prior = gap["previous_primary_timestamp"]
        nxt = gap["next_primary_timestamp"]
        candidate = supplemental[
            (supplemental["timestamp"] > prior) & (supplemental["timestamp"] < nxt)
        ].copy()
        pieces.append(candidate)
        receipts.append(
            {
                "previous_primary_timestamp": pd.Timestamp(prior).isoformat(),
                "next_primary_timestamp": pd.Timestamp(nxt).isoformat(),
                "primary_gap_minutes": gap["gap_minutes"],
                "supplemental_rows": int(len(candidate)),
                "supplemental_first_timestamp": (
                    pd.Timestamp(candidate["timestamp"].min()).isoformat() if len(candidate) else None
                ),
                "supplemental_last_timestamp": (
                    pd.Timestamp(candidate["timestamp"].max()).isoformat() if len(candidate) else None
                ),
                "supplemental_contracts": (
                    sorted(candidate["source_contract"].astype(str).unique().tolist()) if len(candidate) else []
                ),
            }
        )

    fill = pd.concat(pieces, ignore_index=True) if pieces else supplemental.iloc[0:0].copy()
    if fill["timestamp"].duplicated().any():
        raise RuntimeError("material-gap supplemental rows contain duplicate timestamps")
    deep_ts = pd.Index(deep["timestamp"])
    if fill["timestamp"].isin(deep_ts).any():
        raise RuntimeError("material-gap supplement overlaps a primary-owned timestamp")

    union = pd.concat([deep, fill], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if union["timestamp"].duplicated().any():
        raise RuntimeError("material-gap union contains duplicate timestamps")
    if not union["timestamp"].is_monotonic_increasing:
        raise RuntimeError("material-gap union timestamps are not monotonic")
    return union, fill, receipts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--licensed-aggregate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    deep, licensed, sources = base.build_sources(args.deep_root, args.licensed_aggregate)
    overlap = base.overlap_audit(deep, licensed)
    if not overlap["pass"]:
        raise RuntimeError("licensed fill source failed frozen overlap compatibility gate")

    union, fill, material_gap_receipts = build_material_gap_union(deep, licensed)
    bars12, dropped_mixed_bars = base.union_12min_bars(union)
    gap_gate, union_gaps = base._post_union_gap_gate(union)
    bars_cov = coverage(bars12)
    q4_gate = {
        str(year): int(bars_cov.get(f"{year}Q4", {}).get("rows", 0)) >= base.MIN_Q4_12MIN_BARS
        for year in (2023, 2024, 2025)
    }
    coverage_gate = {
        "post_union_gap": bool(gap_gate["pass"]),
        "q4_12min_support_2023_2025": bool(all(q4_gate.values())),
    }
    accepted = bool(overlap["pass"] and all(coverage_gate.values()))

    result = {
        "schema": "foundry.deep_mnq_licensed_gap_fill.v2",
        "research_only": True,
        "promotion_authority": False,
        "authority_issue": "XoticHaze/research-foundry#78",
        "parent_contract": "research/deep_mnq_licensed_gap_fill_contract_20260901.json",
        "integrity_amendment": AMENDMENT,
        "sources": sources,
        "union_policy": (
            "deep primary everywhere; supplemental rows admitted only strictly inside contiguous "
            ">14-day deep-primary gaps; no incidental-minute patching; no interpolation/back-adjustment/price modification"
        ),
        "overlap_audit": overlap,
        "primary_material_gaps": material_gap_receipts,
        "primary_rows": int(len(deep)),
        "licensed_selected_rows": int(len(licensed)),
        "filled_minutes": int(len(fill)),
        "filled_minutes_by_quarter": base._quarter_counts(fill),
        "union_rows": int(len(union)),
        "union_rows_by_quarter": base._quarter_counts(union),
        "source_rows": {
            str(k): int(v) for k, v in union["source_dataset"].value_counts().sort_index().items()
        },
        "source_contract_count": int(union["source_contract"].nunique()),
        "seam_audit": base._seam_audit(union),
        "post_union_gap_gate": gap_gate,
        "post_union_largest_gaps": union_gaps[:20],
        "bars_12min": {
            "rows": int(len(bars12)),
            "dropped_mixed_source_or_contract_bars": dropped_mixed_bars,
            "quarter_coverage": bars_cov,
            "q4_minimum_rows": base.MIN_Q4_12MIN_BARS,
            "q4_gate": q4_gate,
        },
        "coverage_gate": coverage_gate,
        "union_fingerprint_sha256": base._union_fingerprint(union),
        "accepted_for_full_calendar_revalidation": accepted,
        "next_action": (
            "rerun earned MNQ hypotheses against this union receipt before any full-calendar claim"
            if accepted
            else "fill only residual material gaps from same licensed archive partitions or already-owned MM/IBKR history"
        ),
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("DEEP_MNQ_LICENSED_MATERIAL_GAP_FILL=" + ("PASS" if accepted else "FAIL"))
    print("FILLED_MINUTES=" + str(len(fill)))
    print("FILLED_BY_QUARTER=" + json.dumps(result["filled_minutes_by_quarter"], sort_keys=True))
    print("PRIMARY_MATERIAL_GAPS=" + json.dumps(material_gap_receipts, sort_keys=True))
    print("POST_UNION_GAP_GATE=" + json.dumps(gap_gate, sort_keys=True))
    print("Q4_GATE=" + json.dumps(q4_gate, sort_keys=True))
    print("UNION_FINGERPRINT_SHA256=" + result["union_fingerprint_sha256"])
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
