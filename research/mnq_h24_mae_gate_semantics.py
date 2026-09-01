from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

FLOAT_ATOL = 1e-12


def _ge(value: float | int | None, threshold: float | int) -> bool:
    if value is None:
        return False
    value_f = float(value)
    threshold_f = float(threshold)
    if not math.isfinite(value_f) or not math.isfinite(threshold_f):
        return False
    return value_f >= threshold_f or math.isclose(
        value_f,
        threshold_f,
        rel_tol=0.0,
        abs_tol=FLOAT_ATOL,
    )


def _normalize_side(side: dict, frozen: dict) -> dict:
    out = copy.deepcopy(side)
    out["original_predeclared_statistical_gate"] = copy.deepcopy(
        side["predeclared_statistical_gate"]
    )
    bins = side["risk_stratification"]
    phases = side["nonoverlap_h24_phase_robustness"]
    checks = {
        "full_spearman_ge_0p15": _ge(
            side["full_spearman"], frozen["full_spearman_minimum"]
        ),
        "bin_mean_rank_spearman_ge_0p80": _ge(
            bins["realized_mean_rank_spearman"],
            frozen["bin_mean_rank_spearman_minimum"],
        ),
        "q4_realized_mean_gt_q1": bins["q4_minus_q1_realized_mean"] > 0,
        "q4_minus_q1_tail_rate_ge_0p10": _ge(
            bins["q4_minus_q1_tail_exceedance_rate"],
            frozen["q4_minus_q1_tail_rate_minimum"],
        ),
        "positive_phase_spearman_ge_18_of_24": phases["positive_spearman_phases"]
        >= frozen["minimum_positive_nonoverlap_phases_of_24"],
        "positive_phase_q4_minus_q1_ge_18_of_24": phases[
            "positive_q4_minus_q1_phases"
        ]
        >= frozen["minimum_positive_nonoverlap_phases_of_24"],
    }
    out["predeclared_statistical_gate"] = {
        "pass": bool(all(checks.values())),
        "checks": checks,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = json.loads(args.input.read_text())
    if raw.get("schema") != "foundry.mnq_h24_mae_yahoo_julaug_transfer.v1":
        raise RuntimeError(f"unexpected receipt schema: {raw.get('schema')}")
    frozen = raw["predeclared_gate"]

    out = copy.deepcopy(raw)
    raw_receipt_sha256 = out.pop("receipt_sha256", None)
    out["raw_receipt_sha256"] = raw_receipt_sha256
    out["gate_semantics"] = {
        "schema": "foundry.inclusive_numeric_gate_semantics.v1",
        "floating_boundary_abs_tolerance": FLOAT_ATOL,
        "rel_tolerance": 0.0,
        "scope": "inclusive floating-point >= comparisons only; strict positivity and integer-count gates unchanged",
        "reason": "avoid IEEE-754 representation rejecting a mathematically exact frozen boundary such as 0.8 serialized as 0.7999999999999999",
        "thresholds_changed": False,
        "metrics_changed": False,
    }
    out["sides"] = {
        name: _normalize_side(side, frozen) for name, side in raw["sides"].items()
    }
    material = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    out["receipt_sha256"] = hashlib.sha256(material).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("MNQ_H24_MAE_GATE_SEMANTICS=NORMALIZED")
    print("FLOAT_ATOL=" + str(FLOAT_ATOL))
    print("RAW_RECEIPT_SHA256=" + str(raw_receipt_sha256))
    for name, side in out["sides"].items():
        print(
            f"{name.upper()}_GATE_ORIGINAL={side['original_predeclared_statistical_gate']['pass']}"
        )
        print(f"{name.upper()}_GATE_NORMALIZED={side['predeclared_statistical_gate']['pass']}")
    print("NORMALIZED_RECEIPT_SHA256=" + out["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
