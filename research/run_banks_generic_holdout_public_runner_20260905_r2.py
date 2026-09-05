from __future__ import annotations

"""R2 public runner: preserve the exact accepted calendars before reusing R1 semantics."""

import argparse
import json
from pathlib import Path

import pandas as pd

import run_banks_generic_holdout_public_runner_20260905 as base


def _prime_cache(symbols: tuple[str, ...], cutoff_text: str) -> None:
    cutoff = pd.Timestamp(cutoff_text)
    base._CACHE.clear()
    loaded = {}
    for symbol in symbols:
        frame = base._load(symbol)
        bounded = frame.loc[frame.timestamp <= cutoff].reset_index(drop=True)
        if bounded.empty or bounded.iloc[-1].timestamp != cutoff:
            raise RuntimeError(
                f"{symbol}: expected exact frozen cutoff {cutoff.isoformat()} "
                f"but last available bounded row is "
                f"{None if bounded.empty else bounded.iloc[-1].timestamp.isoformat()}"
            )
        loaded[symbol] = bounded
    base._CACHE.clear()
    base._CACHE.update(loaded)


def run(contract_path: Path, output_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    generic = contract["generic_prior"]
    if tuple(generic["training_universe"]) != base.TRAIN:
        raise RuntimeError("training universe drift")
    if list(generic["features"]) != base.FEATURES:
        raise RuntimeError("feature drift")

    parity_contract = contract["parity_control"]
    parity_symbols = (*base.TRAIN, *tuple(parity_contract["holdout"]), "QQQ", str(parity_contract["benchmark"]))
    _prime_cache(parity_symbols, parity_contract["common_cutoff"])
    parity_result = base._evaluate_family(
        tuple(parity_contract["holdout"]),
        str(parity_contract["benchmark"]),
    )
    parity_checks = base._assert_homebuilder_parity(parity_result, parity_contract["accepted_result"])

    # The Bank target is loaded only after parity has succeeded. Its calendar is the exact
    # cutoff frozen by the successful #270 development screen, not today's latest source row.
    target = contract["target"]
    target_symbols = (*base.TRAIN, *tuple(target["holdout"]), "QQQ", str(target["benchmark"]))
    _prime_cache(target_symbols, target["common_cutoff"])
    bank_result = base._evaluate_family(tuple(target["holdout"]), str(target["benchmark"]))

    result = {
        "schema": "foundry.research.public_runner_industry_holdout_result.v2",
        "scientific_authority": contract["scientific_authority"],
        "environment": contract["generic_prior"]["accepted_environment"],
        "parity": {
            "passed": True,
            "common_cutoff": parity_contract["common_cutoff"],
            "checks": parity_checks,
            "result": parity_result,
        },
        "target": {
            "family": "banks",
            "common_cutoff": target["common_cutoff"],
            "result": bank_result,
        },
        "boundaries": {
            **contract["boundaries"],
            "homebuilder_parity_required_before_bank_load": True,
            "homebuilder_parity_passed": True,
            "bank_holdout_target_rows_used_in_training": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.contract), Path(args.output))


if __name__ == "__main__":
    main()
