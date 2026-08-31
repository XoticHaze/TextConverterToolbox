from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from research import mnq_h24_mae_risk_specialist as base
from research.deep_mnq_source_contract import contract_receipt, load_deep

DESCRIPTIVE_MIN_VALID_POLICY_WEEKS = 50


def _output_path(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--output") + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--output is required") from exc


def _valid_week_counts(data: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for policy, summary in data.get("summary", {}).items():
        cost_summary = summary.get("after_1p0pt", {})
        counts[policy] = int(cost_summary.get("weeks", 0))
    return counts


def main() -> int:
    # Preserve the frozen MAE-risk models, targets, OOF threshold, veto rule,
    # costs, and OOS geometry. Correct only the deep timestamp parser.
    base.load_deep = load_deep

    # The corrected trade-week boundaries leave 57 valid policy weeks instead
    # of the original reporting guard's 60. Allow a descriptive receipt to be
    # emitted, but record the original gate explicitly and do NOT treat a
    # below-60 result as promotion evidence.
    original_min = int(base.MIN_VALID_POLICY_WEEKS)
    base.MIN_VALID_POLICY_WEEKS = DESCRIPTIVE_MIN_VALID_POLICY_WEEKS
    rc = base.main()

    path = _output_path(sys.argv)
    data = json.loads(path.read_text())
    counts = _valid_week_counts(data)
    original_gate_pass = bool(counts) and all(v >= original_min for v in counts.values())

    data["deep_timestamp_contract"] = contract_receipt()
    data["timestamp_correction_generation"] = "timefixed_r1"
    data["original_min_valid_policy_weeks"] = original_min
    data["timefixed_descriptive_min_valid_policy_weeks"] = DESCRIPTIVE_MIN_VALID_POLICY_WEEKS
    data["valid_policy_week_counts"] = counts
    data["original_min_valid_policy_week_gate_pass"] = original_gate_pass
    data["timefixed_evidence_status"] = (
        "original_reporting_gate_pass"
        if original_gate_pass
        else "descriptive_only_below_original_60_week_reporting_gate"
    )
    data.pop("receipt_sha256", None)
    material = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print("DEEP_TIMESTAMP_CONTRACT=PASS")
    print("ORIGINAL_MAE_WEEK_GATE_PASS=" + str(original_gate_pass).lower())
    print("VALID_POLICY_WEEK_COUNTS=" + json.dumps(counts, sort_keys=True))
    print("CORRECTED_RECEIPT_SHA256=" + data["receipt_sha256"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
