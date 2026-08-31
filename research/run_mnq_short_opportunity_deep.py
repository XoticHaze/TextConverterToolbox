from __future__ import annotations

"""Run the offline short-opportunity panel on the pinned deep MNQ research corpus."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_short_opportunity_oos import ShortOOSSpec, evaluate_short_challengers
from research.mnq_short_opportunity_targets import ShortOpportunitySpec, materialize_short_opportunity_targets


def economic_matrix(pred: pd.DataFrame) -> list[dict[str, object]]:
    """Rescore one causal OOS probability stream without refitting models.

    Threshold and point-cost choices do not alter training data or probabilities, so
    fitting the three challengers once is economically identical to refitting them for
    every threshold/cost cell and avoids 12x redundant model training.
    """
    panels: list[dict[str, object]] = []
    if pred.empty:
        return panels
    for threshold in (0.50, 0.55, 0.60, 0.65):
        for cost in (0.5, 1.0, 2.0):
            scored = pred.copy()
            scored["participate"] = scored["p_short"] >= threshold
            scored["gross_points"] = np.where(scored["participate"], scored["gross_points"], 0.0)
            scored["net_points"] = np.where(scored["participate"], scored["gross_points"] - cost, 0.0)
            summary = scored.groupby("challenger", as_index=False).agg(
                oos_rows=("row", "size"),
                signals=("participate", "sum"),
                gross_points=("gross_points", "sum"),
                net_points=("net_points", "sum"),
                mean_net_points=("net_points", "mean"),
            )
            summary["coverage"] = summary["signals"] / summary["oos_rows"]
            for row in summary.to_dict("records"):
                panels.append({"threshold": threshold, "point_cost": cost, **row})
    return panels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    features = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    targets = materialize_short_opportunity_targets(frame, ShortOpportunitySpec(horizon_bars=args.horizon))
    work = frame.join(targets)
    cols = ["timestamp", *features, "short_label", "short_forward_points", "target_resolution_row"]
    work = work[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=features).reset_index(drop=True)

    # Fit each causal challenger only once. Use the lowest matrix threshold and zero
    # scoring cost so every probability/gross outcome needed by later rescoring is kept.
    pred, _ = evaluate_short_challengers(
        work,
        features,
        ShortOOSSpec(min_train_rows=5000, retrain_every=1000, probability_threshold=0.50, point_cost=0.0),
    )
    panels = economic_matrix(pred)

    out = {
        "schema": "foundry.mnq_short_opportunity_deep_oos.v1",
        "research_only": True,
        "short_execution_enabled": False,
        "horizon_bars": args.horizon,
        "feature_count": len(features),
        "bars": len(work),
        "first_timestamp": work["timestamp"].iloc[0].isoformat(),
        "last_timestamp": work["timestamp"].iloc[-1].isoformat(),
        "panels": panels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("MNQ_SHORT_OPPORTUNITY_DEEP_OOS=PASS")
    print(f"PANELS={len(panels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
