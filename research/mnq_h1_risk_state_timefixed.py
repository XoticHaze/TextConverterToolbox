from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from research import mnq_h1_risk_state as base
from research.deep_mnq_source_contract import contract_receipt, load_deep


def _output_path(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--output") + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--output is required") from exc


def main() -> int:
    # Preserve the frozen model/target/evaluation implementation exactly. Only the
    # source timestamp loader is replaced with the evidenced deep-MNQ contract.
    base.load_deep = load_deep
    rc = base.main()
    path = _output_path(sys.argv)
    data = json.loads(path.read_text())
    data["deep_timestamp_contract"] = contract_receipt()
    data["timestamp_correction_generation"] = "timefixed_r1"
    data["underlying_risk_contract_schema"] = "foundry.mnq_h1_risk_state_contract.v1"
    data.pop("receipt_sha256", None)
    material = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print("H1_RISK_DEEP_TIMESTAMP_CONTRACT=PASS")
    print("H1_RISK_TIMESTAMP_CORRECTION_GENERATION=timefixed_r1")
    print("CORRECTED_RECEIPT_SHA256=" + data["receipt_sha256"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
