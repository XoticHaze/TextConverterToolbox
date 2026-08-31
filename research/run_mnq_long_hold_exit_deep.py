from __future__ import annotations

"""Run the long HOLD/EXIT specialist on the pinned deep MNQ research corpus.

Research-only. This evaluates hypothetical already-open long trajectories and does
not authorize runtime exits, StrategySpec changes, broker actions, or promotion.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_long_hold_exit_oos import HoldExitOOSSpec, chronological_long_hold_exit_panel, summarize_hold_exit_panel
from research.mnq_long_hold_exit_targets import LongHoldExitSpec, materialize_long_hold_exit_targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--decision", type=int, default=6)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    features = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))

    # ATR in price points, causal at each decision timestamp.
    prev_close = frame["close"].shift(1)
    tr = np.maximum.reduce([
        (frame["high"] - frame["low"]).to_numpy(dtype=float),
        (frame["high"] - prev_close).abs().to_numpy(dtype=float),
        (frame["low"] - prev_close).abs().to_numpy(dtype=float),
    ])
    frame["atr_points"] = __import__("pandas").Series(tr, index=frame.index).rolling(14, min_periods=14).mean()

    target_spec = LongHoldExitSpec(horizon_bars=args.horizon, decision_bars=args.decision, cost_points=1.0)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    valid = frame[features + ["close", "atr_points"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    frame = frame.loc[valid].reset_index(drop=True)
    targets = targets.loc[valid].reset_index(drop=True)

    results = []
    for threshold in (0.45, 0.50, 0.55, 0.60):
        panel = chronological_long_hold_exit_panel(
            frame,
            targets,
            features,
            target_spec,
            HoldExitOOSSpec(min_train_rows=5000, probability_threshold=threshold),
        )
        summary = summarize_hold_exit_panel(panel)
        learned = panel["learned_realized_points"].to_numpy(dtype=float)
        fixed = panel["fixed_horizon_realized_points"].to_numpy(dtype=float)
        trailing = panel["trailing_realized_points"].to_numpy(dtype=float)
        atr_trailing = panel["atr_trailing_realized_points"].to_numpy(dtype=float)
        results.append({
            "threshold": threshold,
            "rows": int(len(panel)),
            "hold_fraction": float((panel["learned_action"] == "HOLD").mean()),
            "summary": summary.to_dict("records"),
            "learned_minus_fixed_mean_points": float(np.mean(learned - fixed)),
            "learned_minus_trailing_mean_points": float(np.mean(learned - trailing)),
            "learned_minus_atr_trailing_mean_points": float(np.mean(learned - atr_trailing)),
        })

    out = {
        "schema": "foundry.mnq_long_hold_exit_deep_oos.v1",
        "research_only": True,
        "runtime_exit_authority": False,
        "horizon_bars": args.horizon,
        "decision_bars": args.decision,
        "feature_count": len(features),
        "bars": len(frame),
        "first_timestamp": frame["timestamp"].iloc[0].isoformat(),
        "last_timestamp": frame["timestamp"].iloc[-1].isoformat(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("MNQ_LONG_HOLD_EXIT_DEEP_OOS=PASS")
    print(f"CONFIGS={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
