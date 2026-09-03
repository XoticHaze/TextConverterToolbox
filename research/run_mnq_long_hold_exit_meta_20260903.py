from __future__ import annotations

"""Causal conditional specialist over the rejected unconditional MNQ HOLD/EXIT policy.

ATR trailing remains the default. The fixed learned threshold=0.50 policy is used only
when a prior-only Ridge meta-model predicts positive learned-minus-ATR incremental value
from all pre-existing decision-time regime flags. Research only; no runtime authority.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_long_hold_exit_oos import HoldExitOOSSpec, chronological_long_hold_exit_panel
from research.mnq_long_hold_exit_targets import LongHoldExitSpec, materialize_long_hold_exit_targets

DEFAULT_CONTRACT = Path("research/mnq_long_hold_exit_meta_contract_20260903.json")
DEFAULT_OUTPUT = Path("research/results/mnq_long_hold_exit_meta_20260903.json")


def _build_parent_panel(deep_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_deep(deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    features = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))

    prev_close = frame["close"].shift(1)
    tr = np.maximum.reduce([
        (frame["high"] - frame["low"]).to_numpy(dtype=float),
        (frame["high"] - prev_close).abs().to_numpy(dtype=float),
        (frame["low"] - prev_close).abs().to_numpy(dtype=float),
    ])
    frame["atr_points"] = pd.Series(tr, index=frame.index).rolling(14, min_periods=14).mean()

    target_spec = LongHoldExitSpec(horizon_bars=24, decision_bars=6, cost_points=1.0)
    targets = materialize_long_hold_exit_targets(frame, target_spec)
    valid = frame[features + ["close", "atr_points"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    frame = frame.loc[valid].reset_index(drop=True)
    targets = targets.loc[valid].reset_index(drop=True)

    panel = chronological_long_hold_exit_panel(
        frame,
        targets,
        features,
        target_spec,
        HoldExitOOSSpec(
            min_train_rows=5000,
            probability_threshold=0.50,
            refit_interval_rows=250,
        ),
    ).reset_index(drop=True)
    if len(panel) < 100000:
        raise RuntimeError(f"unexpectedly shallow parent OOS panel: {len(panel)}")
    decision_rows = panel["decision_row"].to_numpy(int)
    resolution_rows = panel["target_resolution_row"].to_numpy(int)
    if np.any(np.diff(decision_rows) < 0) or np.any(np.diff(resolution_rows) < 0):
        raise RuntimeError("parent OOS panel chronology is not monotonic")
    return frame, panel


def _fit_model(x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    model.fit(x, y)
    return model


def _apply_meta(frame: pd.DataFrame, panel: pd.DataFrame, contract: dict) -> pd.DataFrame:
    features = list(contract["meta_features"])
    if features != list(REGIME_FEATURES):
        raise RuntimeError("meta feature contract drift")
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing meta features: {missing}")

    decision_rows = panel["decision_row"].to_numpy(int)
    resolution_rows = panel["target_resolution_row"].to_numpy(int)
    x = frame.loc[decision_rows, features].to_numpy(float)
    learned = panel["learned_realized_points"].to_numpy(float)
    atr = panel["atr_trailing_realized_points"].to_numpy(float)
    target = learned - atr

    minimum = int(contract["meta_model"]["minimum_resolved_training_rows"])
    refit_interval = int(contract["meta_model"]["refit_interval_resolved_rows"])
    if minimum != 5000 or refit_interval != 250:
        raise RuntimeError("meta support/refit contract drift")

    predictions = np.full(len(panel), np.nan, dtype=float)
    selected = np.zeros(len(panel), dtype=bool)
    train_rows_used = np.zeros(len(panel), dtype=int)
    cached_model: Pipeline | None = None
    cached_count = 0

    for i, decision_row in enumerate(decision_rows):
        # target_resolution_row is monotonic, so this returns only strictly resolved prior outcomes.
        resolved_count = int(np.searchsorted(resolution_rows, decision_row, side="left"))
        resolved_count = min(resolved_count, i)
        train_rows_used[i] = resolved_count
        if resolved_count < minimum:
            continue
        if cached_model is None or resolved_count - cached_count >= refit_interval:
            cached_model = _fit_model(x[:resolved_count], target[:resolved_count])
            cached_count = resolved_count
        prediction = float(cached_model.predict(x[i : i + 1])[0])
        predictions[i] = prediction
        selected[i] = prediction > 0.0

    out = panel.copy()
    out["meta_predicted_learned_minus_atr_points"] = predictions
    out["meta_train_resolved_rows"] = train_rows_used
    out["meta_supported"] = np.isfinite(predictions)
    out["specialist_selected"] = selected
    out["hybrid_realized_points"] = np.where(selected, learned, atr)
    out["hybrid_minus_atr_points"] = out["hybrid_realized_points"].to_numpy(float) - atr
    out["hybrid_minus_unconditional_learned_points"] = out["hybrid_realized_points"].to_numpy(float) - learned
    out["oos_year"] = pd.to_datetime(out["timestamp"], utc=True).dt.year
    for feature in features:
        out[feature] = frame.loc[decision_rows, feature].to_numpy(float)
    return out


def _year_rows(out: pd.DataFrame) -> list[dict]:
    rows = []
    for year, group in out.groupby("oos_year", sort=True):
        rows.append({
            "year": int(year),
            "rows": int(len(group)),
            "meta_supported_rows": int(group["meta_supported"].sum()),
            "specialist_selected_rows": int(group["specialist_selected"].sum()),
            "specialist_fraction": float(group["specialist_selected"].mean()),
            "hybrid_mean_points": float(group["hybrid_realized_points"].mean()),
            "atr_mean_points": float(group["atr_trailing_realized_points"].mean()),
            "unconditional_learned_mean_points": float(group["learned_realized_points"].mean()),
            "hybrid_minus_atr_mean_points": float(group["hybrid_minus_atr_points"].mean()),
            "hybrid_minus_atr_total_points": float(group["hybrid_minus_atr_points"].sum()),
            "hybrid_minus_unconditional_learned_mean_points": float(group["hybrid_minus_unconditional_learned_points"].mean()),
        })
    return rows


def run(deep_root: Path, contract_path: Path, output_path: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    if not contract.get("frozen_before_meta_policy_outcomes"):
        raise RuntimeError("meta-policy contract is not prospectively frozen")
    if float(contract["base_exit_policies"]["learned_threshold"]) != 0.50:
        raise RuntimeError("learned threshold drift")
    if float(contract["meta_model"]["activation_threshold"]) != 0.0:
        raise RuntimeError("meta activation threshold drift")

    frame, parent_panel = _build_parent_panel(deep_root)
    out = _apply_meta(frame, parent_panel, contract)
    years = _year_rows(out)
    full_years = [int(y) for y in contract["evaluation"]["full_years"]]
    full = [row for row in years if row["year"] in full_years]
    if [row["year"] for row in full] != full_years:
        raise RuntimeError(f"missing full evaluation years: {[row['year'] for row in full]}")

    aggregate_delta = float(out["hybrid_minus_atr_points"].mean())
    full_means = np.asarray([row["hybrid_minus_atr_mean_points"] for row in full], dtype=float)
    positive_years = int(np.sum(full_means > 0.0))
    median_year = float(np.median(full_means))
    selected_total = int(out["specialist_selected"].sum())
    years_with_100 = int(sum(row["specialist_selected_rows"] >= 100 for row in full))
    positive_totals = [max(0.0, float(row["hybrid_minus_atr_total_points"])) for row in full]
    positive_total_sum = float(sum(positive_totals))
    concentration = float(max(positive_totals) / positive_total_sum) if positive_total_sum > 0 else 1.0

    gate = bool(
        aggregate_delta > 0.0
        and positive_years >= 4
        and median_year > 0.0
        and selected_total >= 1000
        and years_with_100 >= 4
        and concentration <= 0.50
    )

    regime_activation = []
    for feature in contract["meta_features"]:
        active = out[out[feature] >= 0.5]
        regime_activation.append({
            "regime": feature,
            "rows": int(len(active)),
            "specialist_selected_rows": int(active["specialist_selected"].sum()),
            "specialist_fraction": float(active["specialist_selected"].mean()) if len(active) else None,
            "hybrid_minus_atr_mean_points": float(active["hybrid_minus_atr_points"].mean()) if len(active) else None,
        })

    payload = {
        "schema": "research.mnq_long_hold_exit_meta_oos.v1",
        "contract": contract,
        "parent_oos_rows": int(len(parent_panel)),
        "meta_supported_rows": int(out["meta_supported"].sum()),
        "specialist_selected_rows": selected_total,
        "specialist_selected_fraction": float(out["specialist_selected"].mean()),
        "hybrid_mean_points": float(out["hybrid_realized_points"].mean()),
        "atr_mean_points": float(out["atr_trailing_realized_points"].mean()),
        "unconditional_learned_mean_points": float(out["learned_realized_points"].mean()),
        "aggregate_hybrid_minus_atr_mean_points": aggregate_delta,
        "aggregate_hybrid_minus_unconditional_learned_mean_points": float(out["hybrid_minus_unconditional_learned_points"].mean()),
        "year_decomposition": years,
        "full_year_positive_count": positive_years,
        "full_year_median_hybrid_minus_atr_mean_points": median_year,
        "full_years_with_100_specialist_decisions": years_with_100,
        "max_positive_full_year_concentration": concentration,
        "regime_activation_diagnostics": regime_activation,
        "gate": {
            "aggregate_positive": aggregate_delta > 0.0,
            "at_least_4_of_6_full_years_positive": positive_years >= 4,
            "full_year_median_positive": median_year > 0.0,
            "specialist_support": selected_total >= 1000,
            "year_support": years_with_100 >= 4,
            "positive_year_concentration": concentration <= 0.50,
            "conditional_specialist_development_evidence": gate,
        },
        "classification": contract["development_gate"]["classification_pass"] if gate else contract["development_gate"]["classification_fail"],
        "boundaries": {
            "manual_regime_selection": False,
            "activation_threshold_search": False,
            "learned_threshold_search": False,
            "year_selection": False,
            "runtime_exit_authority": False,
            "promotion_authority": False,
            "live_trading_change": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    run(args.deep_root, args.contract, args.output)


if __name__ == "__main__":
    main()
