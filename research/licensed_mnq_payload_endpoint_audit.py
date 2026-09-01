from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

EXPECTED = [
    "timestamp", "rtype", "publisher_id", "instrument_id", "open", "high", "low", "close", "volume", "symbol"
]
OUTRIGHT = re.compile(r"^MNQ[FGHJKMNQUVXZ]\d{1,2}$")
SPREAD = re.compile(r"^MNQ[FGHJKMNQUVXZ]\d{1,2}-MNQ[FGHJKMNQUVXZ]\d{1,2}$")
WINDOWS = {
    "gap_2022": (
        pd.Timestamp("2022-09-08T04:00:00Z"),
        pd.Timestamp("2022-11-16T00:00:00Z"),
    ),
    "gap_2025": (
        pd.Timestamp("2025-11-13T00:00:00Z"),
        pd.Timestamp("2025-12-11T05:00:00Z"),
    ),
}


def audit_file(path: Path) -> dict:
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    row = {
        "path": path.as_posix(),
        "name": path.name,
        "columns": columns,
        "expected_schema": columns == EXPECTED,
        "rows": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "outright_rows": 0,
        "spread_rows": 0,
        "unknown_symbol_rows": 0,
        "windows": {key: {"rows": 0, "outright_rows": 0, "spread_rows": 0, "symbols": []} for key in WINDOWS},
    }
    if columns != EXPECTED:
        return row

    symbols_by_window = {key: set() for key in WINDOWS}
    for chunk in pd.read_csv(path, usecols=["timestamp", "symbol"], chunksize=250_000):
        ts = pd.to_datetime(chunk["timestamp"], utc=True, errors="raise")
        symbols = chunk["symbol"].astype(str).str.strip()
        if len(chunk):
            first = ts.iloc[0]
            last = ts.iloc[-1]
            if row["first_timestamp"] is None or first.isoformat() < row["first_timestamp"]:
                row["first_timestamp"] = first.isoformat()
            if row["last_timestamp"] is None or last.isoformat() > row["last_timestamp"]:
                row["last_timestamp"] = last.isoformat()
        outright = symbols.map(lambda value: bool(OUTRIGHT.fullmatch(value)))
        spread = symbols.map(lambda value: bool(SPREAD.fullmatch(value)))
        row["rows"] += int(len(chunk))
        row["outright_rows"] += int(outright.sum())
        row["spread_rows"] += int(spread.sum())
        row["unknown_symbol_rows"] += int((~(outright | spread)).sum())

        for key, (start, end) in WINDOWS.items():
            mask = (ts >= start) & (ts < end)
            if not mask.any():
                continue
            m_out = outright & mask
            m_spread = spread & mask
            state = row["windows"][key]
            state["rows"] += int(mask.sum())
            state["outright_rows"] += int(m_out.sum())
            state["spread_rows"] += int(m_spread.sum())
            symbols_by_window[key].update(symbols.loc[mask].unique().tolist())

    for key in WINDOWS:
        row["windows"][key]["symbols"] = sorted(symbols_by_window[key])
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(path for path in args.root.rglob("*.csv") if path.is_file())
    if not files:
        raise RuntimeError("licensed archive contains no CSV payloads")
    rows = [audit_file(path) for path in files]
    totals = {}
    for key in WINDOWS:
        candidates = [
            {
                "name": row["name"],
                "rows": row["windows"][key]["rows"],
                "outright_rows": row["windows"][key]["outright_rows"],
                "spread_rows": row["windows"][key]["spread_rows"],
                "symbols": row["windows"][key]["symbols"],
            }
            for row in rows
            if row["windows"][key]["rows"] > 0
        ]
        totals[key] = {
            "candidate_files": candidates,
            "files_with_rows": len(candidates),
            "files_with_outright_rows": sum(1 for item in candidates if item["outright_rows"] > 0),
            "total_rows_across_files_including_overlap": sum(item["rows"] for item in candidates),
            "total_outright_rows_across_files_including_overlap": sum(item["outright_rows"] for item in candidates),
        }

    result = {
        "schema": "foundry.licensed_mnq_payload_endpoint_audit.v1",
        "research_only": True,
        "source": "brandenmorris/mnq-1m-q4-2022-q4-2025-ts-ohlcv-sym",
        "version": 1,
        "zip_sha256": "633eb4338a3aa60aedb542b12085acca29ff237c7cadff65442a638466f37667",
        "windows": {
            key: {"start": start.isoformat(), "end_exclusive": end.isoformat()}
            for key, (start, end) in WINDOWS.items()
        },
        "file_count": len(files),
        "files": rows,
        "residual_window_summary": totals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LICENSED_MNQ_PAYLOAD_ENDPOINT_AUDIT=PASS")
    print("FILE_COUNT=" + str(len(files)))
    print("RESIDUAL_WINDOW_SUMMARY=" + json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
