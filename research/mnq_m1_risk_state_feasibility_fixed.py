from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research import mnq_m1_risk_state as base

AMENDMENT = "research/mnq_m1_risk_state_feasibility_amendment_20260901.json"
MIN_ELIGIBLE_QUARTERS = 9
REQUIRED_FRACTION_NUMERATOR = 5
REQUIRED_FRACTION_DENOMINATOR = 6


def required_stable_quarters(eligible_quarters: int) -> int:
    if eligible_quarters < MIN_ELIGIBLE_QUARTERS:
        raise RuntimeError(
            f"insufficient eligible 1Min risk-state quarters: {eligible_quarters} < {MIN_ELIGIBLE_QUARTERS}"
        )
    return int(math.ceil(eligible_quarters * REQUIRED_FRACTION_NUMERATOR / REQUIRED_FRACTION_DENOMINATOR))


def evaluate_horizon(matrix: pd.DataFrame, horizon: int) -> dict:
    quarter_starts = list(pd.date_range(base.TEST_START, base.TEST_END, freq="QS", tz="UTC"))
    folds: list[dict] = []
    eligibility: list[dict] = []

    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        quarter = f"{start.year}Q{((start.month - 1) // 3) + 1}"
        test_mask = (matrix["timestamp"] >= start) & (matrix["timestamp"] < end)
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < base.MIN_TEST_ROWS:
            eligibility.append({
                "quarter": quarter,
                "eligible": False,
                "reason": "minimum_test_rows",
                "candidate_test_rows": int(len(test_idx)),
            })
            continue

        test_start_idx = int(test_idx[0])
        train_end = test_start_idx - horizon
        if train_end < base.MIN_TRAIN_ROWS:
            eligibility.append({
                "quarter": quarter,
                "eligible": False,
                "reason": "minimum_training_rows_after_horizon_purge",
                "candidate_train_rows": int(train_end),
                "candidate_test_rows": int(len(test_idx)),
            })
            continue

        train = matrix.iloc[:train_end]
        test = matrix.iloc[int(test_idx[0]) : int(test_idx[-1] + 1)].copy()
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end)].copy()
        if len(train) < base.MIN_TRAIN_ROWS or len(test) < base.MIN_TEST_ROWS:
            eligibility.append({
                "quarter": quarter,
                "eligible": False,
                "reason": "post_slice_minimum_rows",
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            })
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("1Min risk-state walk-forward chronology violation")

        metrics: dict[str, dict] = {}
        x_train = train[base.FEATURES].to_numpy(float)
        x_test = test[base.FEATURES].to_numpy(float)
        for target in base.TARGETS:
            model = base.model_for(target).fit(x_train, train[target].to_numpy(float))
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            truth = test[target].to_numpy(float)
            cuts = base.train_cutpoints(train_pred)
            metrics[target] = {
                "pearson": base.corr(test_pred, truth, "pearson"),
                "spearman": base.corr(test_pred, truth, "spearman"),
                "predicted_mean": float(np.mean(test_pred)),
                "realized_mean": float(np.mean(truth)),
                "cohorts": base.cohort_metrics(test_pred, truth, cuts),
            }

        fold = {
            "quarter": quarter,
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "metrics": metrics,
        }
        folds.append(fold)
        eligibility.append({
            "quarter": quarter,
            "eligible": True,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        })

    if len(folds) < MIN_ELIGIBLE_QUARTERS:
        raise RuntimeError(
            f"insufficient 1Min risk-state quarters for H{horizon}: {len(folds)} < {MIN_ELIGIBLE_QUARTERS}"
        )

    required = required_stable_quarters(len(folds))
    return {
        "folds": folds,
        "summary": base.summarize(folds),
        "eligibility": eligibility,
        "eligible_quarters": int(len(folds)),
        "required_stable_quarters": required,
        "required_stable_fraction": REQUIRED_FRACTION_NUMERATOR / REQUIRED_FRACTION_DENOMINATOR,
    }


def decision(summary: dict, target: str) -> dict:
    row = summary[target]
    eligible = int(row["quarters"])
    required = required_stable_quarters(eligible)

    if target in ("long_mae_z", "short_mae_z"):
        checks = {
            "eligible_quarters_at_least_9": eligible >= MIN_ELIGIBLE_QUARTERS,
            "positive_spearman_required_5of6_fraction": row["spearman_positive_quarters"] >= required,
            "median_spearman_at_least_0p15": row["spearman_median"] >= 0.15,
            "q5_gt_q1_required_5of6_fraction": row["q5_gt_q1_quarters"] >= required,
        }
    else:
        checks = {
            "eligible_quarters_at_least_9": eligible >= MIN_ELIGIBLE_QUARTERS,
            "positive_spearman_required_5of6_fraction": row["spearman_positive_quarters"] >= required,
            "median_spearman_at_least_0p20": row["spearman_median"] >= 0.20,
        }

    return {
        "action": "advance_to_separate_fast_risk_consumer" if all(checks.values()) else "reject_as_fast_risk_state_candidate",
        "eligible_quarters": eligible,
        "required_stable_quarters": required,
        "required_stable_fraction": REQUIRED_FRACTION_NUMERATOR / REQUIRED_FRACTION_DENOMINATOR,
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = base.load_deep(args.deep_root)
    stitched = base.stitch_deep(raw, base.deep_roll_schedule(raw))

    horizons: dict[str, dict] = {}
    decisions: dict[str, dict] = {}
    feasibility: dict[str, dict] = {}

    for horizon in base.HORIZONS:
        matrix = base.build_matrix(stitched, horizon)
        result = evaluate_horizon(matrix, horizon)
        horizons[str(horizon)] = result
        decisions[str(horizon)] = {
            target: decision(result["summary"], target) for target in base.TARGETS
        }
        feasibility[str(horizon)] = {
            "eligible_quarters": result["eligible_quarters"],
            "required_stable_quarters": result["required_stable_quarters"],
            "eligibility": result["eligibility"],
        }
        print("HORIZON", horizon, json.dumps(result["summary"], sort_keys=True))
        print("FEASIBILITY", horizon, json.dumps(feasibility[str(horizon)], sort_keys=True))
        print("DECISION", horizon, json.dumps(decisions[str(horizon)], sort_keys=True))

    result = {
        "schema": "foundry.mnq_m1_risk_state_feasibility_fixed.v1",
        "research_only": True,
        "promotion_authority": False,
        "parent_contract": "research/mnq_m1_risk_state_contract_20260901.json",
        "feasibility_amendment": AMENDMENT,
        "source": f"mbytes21/MNQ_DATA@{base.SOURCE_COMMIT}",
        "deep_timestamp_contract": base.contract_receipt(),
        "bar_minutes": 1,
        "horizons": horizons,
        "decisions": decisions,
        "feasibility": feasibility,
        "feature_contract": base.FEATURES,
        "protocol": "parent quarterly expanding walk-forward 2023-2025 with full-horizon purge and unchanged 150k/30k row minima; only impossible absolute 10-of-12 fold count amended to the same 5/6 stability proportion over eligible folds",
        "next_authority": "fast risk-state evidence only; any timing, stop, admission, sizing or 12Min consumer requires a separately frozen economic contract",
        "nq_boundary": "NQ excluded from model inputs and threshold fitting; reserved for later independent validation after a specific MNQ fast-risk consumer survives discovery",
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_M1_RISK_STATE_FEASIBILITY_FIXED=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
