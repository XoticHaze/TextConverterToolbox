from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATE_RE = re.compile(r"^(\d{8})\.Last\.csv$")


def quarter(date: str) -> str:
    year = int(date[:4])
    month = int(date[4:6])
    return f"{year}Q{((month - 1) // 3) + 1}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    quarter_files: Counter[str] = Counter()
    quarter_contracts: dict[str, set[str]] = defaultdict(set)
    contract_dates: dict[str, list[str]] = defaultdict(list)
    dates: list[str] = []

    for path in sorted(args.deep_root.glob("MNQ */*.Last.csv")):
        match = DATE_RE.match(path.name)
        if not match:
            continue
        date = match.group(1)
        q = quarter(date)
        contract = path.parent.name
        quarter_files[q] += 1
        quarter_contracts[q].add(contract)
        contract_dates[contract].append(date)
        dates.append(date)

    if not dates:
        raise RuntimeError(f"no MNQ Last.csv files under {args.deep_root}")

    years = sorted({int(d[:4]) for d in dates})
    by_year = {}
    for year in years:
        by_year[str(year)] = {
            f"Q{i}": {
                "last_csv_files": int(quarter_files.get(f"{year}Q{i}", 0)),
                "contracts": sorted(quarter_contracts.get(f"{year}Q{i}", set())),
            }
            for i in range(1, 5)
        }

    result = {
        "schema": "foundry.deep_mnq_filename_coverage.v1",
        "research_only": True,
        "last_csv_files": int(len(dates)),
        "first_file_date": min(dates),
        "last_file_date": max(dates),
        "quarter_file_counts": dict(sorted(quarter_files.items())),
        "by_year": by_year,
        "contract_file_coverage": {
            contract: {
                "last_csv_files": len(ds),
                "first_file_date": min(ds),
                "last_file_date": max(ds),
                "quarters": sorted({quarter(d) for d in ds}),
            }
            for contract, ds in sorted(contract_dates.items())
        },
        "q4_zero_file_years_with_q1_q3_files": [
            str(year)
            for year in years
            if sum(quarter_files.get(f"{year}Q{i}", 0) for i in (1, 2, 3)) > 0
            and quarter_files.get(f"{year}Q4", 0) == 0
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DEEP_MNQ_FILENAME_COVERAGE=PASS")
    print("Q4_ZERO_FILE_YEARS=" + json.dumps(result["q4_zero_file_years_with_q1_q3_files"]))
    print("BY_YEAR=" + json.dumps(by_year, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
