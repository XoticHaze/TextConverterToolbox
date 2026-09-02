from __future__ import annotations

"""Temporal robustness diagnostic for the frozen Packet C short-opportunity panel.

This consumes the exact parent model/target/feature/probability stream semantics and
reports every predeclared threshold x cost cell by calendar year. It does not select
or tune a threshold and does not authorize short execution.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_short_opportunity_oos import ShortOOSSpec
from research.mnq_short_opportunity_oos_batched import evaluate_short_challengers_batched
from research.mnq_short_opportunity_targets import ShortOpportunitySpec, materialize_short_opportunity_targets

THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
COSTS = (0.5, 1.0, 2.0)
FULL_YEARS = tuple(range(2020, 2026))
PARTIAL_YEARS = (2019, 2026)
MIN_SIGNALS_PER_YEAR = 100


def _cell_classification(
    aggregate_net_per_signal: float | None,
    supported_full_years: int,
    positive_supported_full_years: int,
    strongest_positive_year_share: float | None,
) -> str:
    if supported_full_years < 5:
        return "INSUFFICIENT_SUPPORT"
    if aggregate_net_per_signal is None or aggregate_net_per_signal <= 0:
        return "ECONOMICALLY_NEGATIVE"
    concentration_ok = strongest_positive_year_share is not None and strongest_positive_year_share <= 0.60
    if positive_supported_full_years >= 4 and concentration_ok:
        return "TEMPORALLY_ROBUST"
    return "TEMPORALLY_CONCENTRATED"


def temporal_matrix(pred: pd.DataFrame, timestamps: pd.Series) -> list[dict[str, object]]:
    if pred.empty:
        return []
    out = pred.copy()
    row_ids = pd.to_numeric(out["row"], errors="raise").astype(int).to_numpy()
    mapped = pd.to_datetime(timestamps.iloc[row_ids].to_numpy(), utc=True, errors="raise")
    out["timestamp"] = mapped
    out["year"] = out["timestamp"].dt.year.astype(int)

    cells: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        for cost in COSTS:
            for challenger in sorted(out["challenger"].astype(str).unique()):
                base = out.loc[out["challenger"].astype(str) == challenger].copy()
                selected = base.loc[pd.to_numeric(base["p_short"], errors="raise") >= threshold].copy()
                gross = pd.to_numeric(selected["gross_points"], errors="raise").astype(float)
                selected["net_at_cost"] = gross - float(cost)
                signals = int(len(selected))
                gross_points = float(gross.sum()) if signals else 0.0
                net_points = float(selected["net_at_cost"].sum()) if signals else 0.0
                aggregate_net_per_signal = (net_points / signals) if signals else None

                years: list[dict[str, object]] = []
                supported: list[dict[str, object]] = []
                for year in sorted(set(FULL_YEARS + PARTIAL_YEARS)):
                    yr = selected.loc[selected["year"] == year]
                    yr_signals = int(len(yr))
                    yr_net = float(yr["net_at_cost"].sum()) if yr_signals else 0.0
                    yr_row = {
                        "year": int(year),
                        "full_year_gate_member": bool(year in FULL_YEARS),
                        "signals": yr_signals,
                        "net_points": yr_net,
                        "net_points_per_signal": (yr_net / yr_signals) if yr_signals else None,
                        "meets_minimum_support": bool(year in FULL_YEARS and yr_signals >= MIN_SIGNALS_PER_YEAR),
                    }
                    years.append(yr_row)
                    if yr_row["meets_minimum_support"]:
                        supported.append(yr_row)

                positive_supported = [r for r in supported if float(r["net_points_per_signal"]) > 0]
                positive_net_total = sum(max(0.0, float(r["net_points"])) for r in supported)
                strongest_positive_year_share = (
                    max(max(0.0, float(r["net_points"])) for r in supported) / positive_net_total
                    if positive_net_total > 0 and supported else None
                )
                classification = _cell_classification(
                    aggregate_net_per_signal,
                    len(supported),
                    len(positive_supported),
                    strongest_positive_year_share,
                )
                cells.append({
                    "challenger": challenger,
                    "threshold": float(threshold),
                    "point_cost": float(cost),
                    "signals": signals,
                    "gross_points": gross_points,
                    "net_points": net_points,
                    "net_points_per_signal": aggregate_net_per_signal,
                    "supported_full_years": len(supported),
                    "positive_supported_full_years": len(positive_supported),
                    "strongest_positive_full_year_share": strongest_positive_year_share,
                    "classification": classification,
                    "years": years,
                })
    return cells


def family_summary(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    families = sorted({str(c["challenger"]) for c in cells})
    rows: list[dict[str, object]] = []
    for challenger in families:
        robust_thresholds: list[float] = []
        for threshold in THRESHOLDS:
            statuses = {
                float(c["point_cost"]): str(c["classification"])
                for c in cells
                if c["challenger"] == challenger and float(c["threshold"]) == threshold
            }
            if statuses.get(1.0) == "TEMPORALLY_ROBUST" and statuses.get(2.0) == "TEMPORALLY_ROBUST":
                robust_thresholds.append(float(threshold))
        rows.append({
            "challenger": challenger,
            "temporal_specialist_evidence": bool(robust_thresholds),
            "robust_thresholds_at_both_1_and_2_point_costs": robust_thresholds,
            "threshold_selected": None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    features = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    targets = materialize_short_opportunity_targets(frame, ShortOpportunitySpec(horizon_bars=24))
    work = frame.join(targets)
    cols = ["timestamp", *features, "short_label", "short_forward_points", "target_resolution_row"]
    work = work[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=features).reset_index(drop=True)

    pred, _ = evaluate_short_challengers_batched(
        work,
        features,
        ShortOOSSpec(min_train_rows=5000, retrain_every=1000, probability_threshold=0.50, point_cost=0.0),
    )
    cells = temporal_matrix(pred, work["timestamp"])
    families = family_summary(cells)

    payload = {
        "schema": "foundry.mnq_short_opportunity_temporal_robustness.v1",
        "research_only": True,
        "short_execution_enabled": False,
        "parent_scientific_head": "0c4fc75517743a1f316f2d66a25a07db74818dee",
        "parent_run": 33543883300,
        "contract": {
            "thresholds": list(THRESHOLDS),
            "point_costs": list(COSTS),
            "full_years": list(FULL_YEARS),
            "partial_years_descriptive_only": list(PARTIAL_YEARS),
            "minimum_signals_per_year": MIN_SIGNALS_PER_YEAR,
            "threshold_selection": False,
            "refit_by_threshold_or_cost": False,
        },
        "bars": int(len(work)),
        "oos_prediction_rows": int(len(pred)),
        "first_timestamp": work["timestamp"].iloc[0].isoformat(),
        "last_timestamp": work["timestamp"].iloc[-1].isoformat(),
        "cells": cells,
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("MNQ_SHORT_TEMPORAL_ROBUSTNESS=PASS")
    for row in families:
        print(f"FAMILY={row['challenger']} TEMPORAL_SPECIALIST_EVIDENCE={row['temporal_specialist_evidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
