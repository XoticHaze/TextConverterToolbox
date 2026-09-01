from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.mnq_external_transfer_validation import deep_bars, deep_roll_schedule, stitch_deep

SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"


def quarter_label(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    return t.dt.year.astype(str) + "Q" + (((t.dt.month - 1) // 3) + 1).astype(str)


def coverage(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> dict[str, dict]:
    if frame.empty:
        return {}
    work = frame.copy()
    work["_quarter"] = quarter_label(work[timestamp_col])
    work["_session"] = pd.to_datetime(work[timestamp_col], utc=True).dt.strftime("%Y-%m-%d")
    out: dict[str, dict] = {}
    for q, group in work.groupby("_quarter", sort=True):
        ts = pd.to_datetime(group[timestamp_col], utc=True)
        out[str(q)] = {
            "rows": int(len(group)),
            "unique_timestamps": int(ts.nunique()),
            "unique_sessions": int(group["_session"].nunique()),
            "first_timestamp": ts.min().isoformat(),
            "last_timestamp": ts.max().isoformat(),
        }
    return out


def contract_coverage(raw: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for symbol, group in raw.groupby("symbol", sort=True):
        ts = pd.to_datetime(group["timestamp"], utc=True)
        out[str(symbol)] = {
            "rows": int(len(group)),
            "unique_timestamps": int(ts.nunique()),
            "first_timestamp": ts.min().isoformat(),
            "last_timestamp": ts.max().isoformat(),
            "quarters": sorted(quarter_label(ts).unique().tolist()),
        }
    return out


def selected_contracts_by_quarter(stitched: pd.DataFrame) -> dict[str, dict[str, int]]:
    work = stitched.copy()
    work["_quarter"] = quarter_label(work["timestamp"])
    out: dict[str, dict[str, int]] = {}
    for q, group in work.groupby("_quarter", sort=True):
        counts = group["symbol"].value_counts().sort_index()
        out[str(q)] = {str(k): int(v) for k, v in counts.items()}
    return out


def largest_gaps(stitched: pd.DataFrame, limit: int = 30) -> list[dict]:
    ts = pd.to_datetime(stitched["timestamp"], utc=True).sort_values().drop_duplicates().reset_index(drop=True)
    if len(ts) < 2:
        return []
    diff = ts.diff()
    order = np.argsort(diff.fillna(pd.Timedelta(0)).to_numpy())[::-1][:limit]
    rows: list[dict] = []
    for i in order:
        if i <= 0:
            continue
        rows.append({
            "previous_timestamp": ts.iloc[int(i - 1)].isoformat(),
            "next_timestamp": ts.iloc[int(i)].isoformat(),
            "gap_minutes": float(diff.iloc[int(i)] / pd.Timedelta(minutes=1)),
            "previous_quarter": str(quarter_label(pd.Series([ts.iloc[int(i - 1)]]))[0]),
            "next_quarter": str(quarter_label(pd.Series([ts.iloc[int(i)]]))[0]),
        })
    return rows


def q4_comparison(raw_cov: dict[str, dict], stitched_cov: dict[str, dict], bar_cov: dict[str, dict]) -> dict:
    years = sorted({int(k[:4]) for k in raw_cov if len(k) >= 6})
    rows: dict[str, dict] = {}
    for year in years:
        qkeys = [f"{year}Q{i}" for i in range(1, 5)]
        if not any(k in raw_cov for k in qkeys):
            continue
        rows[str(year)] = {
            "raw_unique_timestamps": {k[-2:]: int(raw_cov.get(k, {}).get("unique_timestamps", 0)) for k in qkeys},
            "stitched_rows": {k[-2:]: int(stitched_cov.get(k, {}).get("rows", 0)) for k in qkeys},
            "bars_12min": {k[-2:]: int(bar_cov.get(k, {}).get("rows", 0)) for k in qkeys},
        }

    raw_q4_zero = []
    stitched_q4_zero_despite_raw = []
    for year, rec in rows.items():
        q123_raw = sum(rec["raw_unique_timestamps"][f"Q{i}"] for i in (1, 2, 3))
        q4_raw = rec["raw_unique_timestamps"]["Q4"]
        q4_stitched = rec["stitched_rows"]["Q4"]
        if q123_raw > 0 and q4_raw == 0:
            raw_q4_zero.append(year)
        if q4_raw > 0 and q4_stitched == 0:
            stitched_q4_zero_despite_raw.append(year)

    return {
        "by_year": rows,
        "raw_q4_zero_despite_q1_q3_rows_years": raw_q4_zero,
        "stitched_q4_zero_despite_raw_q4_rows_years": stitched_q4_zero_despite_raw,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    schedule = deep_roll_schedule(raw)
    stitched = stitch_deep(raw, schedule)
    bars12 = deep_bars(stitched)

    raw_cov = coverage(raw)
    stitched_cov = coverage(stitched)
    bar_cov = coverage(bars12)
    q4 = q4_comparison(raw_cov, stitched_cov, bar_cov)

    result = {
        "schema": "foundry.deep_mnq_calendar_coverage_audit.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "deep_timestamp_contract": contract_receipt(),
        "raw": {
            "rows": int(len(raw)),
            "unique_timestamps": int(pd.to_datetime(raw["timestamp"], utc=True).nunique()),
            "quarter_coverage": raw_cov,
            "contract_coverage": contract_coverage(raw),
        },
        "roll_schedule": {
            "rows": int(len(schedule)),
            "first_session": str(schedule["session"].min()) if len(schedule) else None,
            "last_session": str(schedule["session"].max()) if len(schedule) else None,
            "selected_contract_counts": {str(k): int(v) for k, v in schedule["selected_contract"].value_counts().sort_index().items()},
        },
        "stitched": {
            "rows": int(len(stitched)),
            "quarter_coverage": stitched_cov,
            "selected_contracts_by_quarter": selected_contracts_by_quarter(stitched),
            "largest_timestamp_gaps": largest_gaps(stitched),
        },
        "bars_12min": {
            "rows": int(len(bars12)),
            "quarter_coverage": bar_cov,
        },
        "q4_comparison": q4,
        "classification": (
            "roll_or_stitch_drops_existing_q4"
            if q4["stitched_q4_zero_despite_raw_q4_rows_years"]
            else (
                "raw_source_has_years_with_zero_q4"
                if q4["raw_q4_zero_despite_q1_q3_rows_years"]
                else "q4_not_zero; inspect relative coverage ratios"
            )
        ),
        "next_authority": "source integrity evidence only; downstream model interpretation must follow this coverage result",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("DEEP_MNQ_CALENDAR_COVERAGE_AUDIT=PASS")
    print("CLASSIFICATION=" + result["classification"])
    print("Q4_COMPARISON=" + json.dumps(q4, sort_keys=True))
    print("LARGEST_GAPS=" + json.dumps(result["stitched"]["largest_timestamp_gaps"][:10], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
