from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.licensed_mnq_expanded_validation import _bars, _load_aggregate, _roll_schedule, _stitch
from research.licensed_mnq_trust_gate import LOCAL_STATE, _base_model, _folds, _gate_model, _metric

HORIZON = 12


def _thresholds(prob: np.ndarray, base: np.ndarray) -> dict[int, dict[str, float]]:
    out = {}
    for cls in (0, 1):
        values = prob[base == cls]
        if len(values) < 500:
            raise RuntimeError(f"insufficient discovery gate rows for base class {cls}: {len(values)}")
        out[cls] = {"low": float(np.quantile(values, 0.25)), "high": float(np.quantile(values, 0.65))}
    return out


def _policy(y: np.ndarray, base: np.ndarray, prob: np.ndarray, thresholds: dict[int, dict[str, float]]) -> dict:
    low = np.array([thresholds[int(cls)]["low"] for cls in base], dtype=float)
    high = np.array([thresholds[int(cls)]["high"] for cls in base], dtype=float)
    invert = prob <= low
    trust = prob >= high
    selected = invert | trust
    result = {
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "coverage": float(selected.mean()),
        "trust_coverage": float(trust.mean()),
        "invert_coverage": float(invert.mean()),
        "selected_base_class_counts": {str(cls): int((selected & (base == cls)).sum()) for cls in (0, 1)},
        "trust_base_class_counts": {str(cls): int((trust & (base == cls)).sum()) for cls in (0, 1)},
        "invert_base_class_counts": {str(cls): int((invert & (base == cls)).sum()) for cls in (0, 1)},
    }
    if trust.sum() >= 100:
        result["trust"] = _metric(y[trust], base[trust])
    if invert.sum() >= 100:
        result["invert"] = _metric(y[invert], 1 - base[invert])
    if selected.sum() >= 200:
        pred = base.copy(); pred[invert] = 1 - pred[invert]
        result["selected"] = _metric(y[selected], pred[selected])
    return result


def _quarter_slices(hold: pd.DataFrame, prob: np.ndarray, thresholds: dict[int, dict[str, float]]) -> dict:
    node = hold[["timestamp", "truth", "base_pred"]].copy(); node["prob"] = prob
    node["period"] = node["timestamp"].dt.strftime("%YQ") + (((node["timestamp"].dt.month - 1) // 3) + 1).astype(str)
    out = {}
    for period, part in node.groupby("period", sort=True):
        if len(part) < 250:
            continue
        out[period] = {"rows": int(len(part)), "base": _metric(part["truth"].to_numpy(int), part["base_pred"].to_numpy(int)),
                       "policy": _policy(part["truth"].to_numpy(int), part["base_pred"].to_numpy(int), part["prob"].to_numpy(float), thresholds)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--aggregate", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    raw = _load_aggregate(args.aggregate); stitched = _stitch(raw, _roll_schedule(raw)); bars = _bars(stitched); frame = _add_features(bars)
    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES)); base_features = list(dict.fromkeys(expanded + REGIME_FEATURES))
    target = "target_dir_h12"
    gate_modes = {
        "regime_only": list(REGIME_FEATURES),
        "local_state": list(dict.fromkeys(REGIME_FEATURES + LOCAL_STATE)),
        "full_state": base_features,
    }
    all_gate = list(dict.fromkeys(base_features + LOCAL_STATE))
    work = frame[list(dict.fromkeys(["timestamp", *base_features, *all_gate, target]))].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    folds = _folds(len(work), HORIZON); X = work[base_features].to_numpy(float); y = work[target].astype(int).to_numpy()

    parts = []
    for fold, (s, e, ts, te) in enumerate(folds[:3]):
        m = _base_model().fit(X[s:e], y[s:e]); pred = m.predict(X[ts:te]).astype(int); up = m.predict_proba(X[ts:te])[:, 1]
        part = work.iloc[ts:te][["timestamp", *all_gate]].copy(); part["base_pred"] = pred; part["base_confidence"] = np.abs(up - .5) * 2
        part["correct"] = (pred == y[ts:te]).astype(int); part["truth"] = y[ts:te]; part["fold"] = fold; parts.append(part)
    meta = pd.concat(parts, ignore_index=True)

    _, e, ts, te = folds[3]; m = _base_model().fit(X[:e], y[:e]); pred = m.predict(X[ts:te]).astype(int); up = m.predict_proba(X[ts:te])[:, 1]
    hold = work.iloc[ts:te][["timestamp", *all_gate]].copy(); hold["base_pred"] = pred; hold["base_confidence"] = np.abs(up - .5) * 2
    hold["correct"] = (pred == y[ts:te]).astype(int); hold["truth"] = y[ts:te]

    result = {"schema": "foundry.licensed_mnq_class_balanced_gate.v1", "research_only": True, "promotion_authority": False,
              "source_dataset": "brandenmorris/mnq-1m-q4-2022-q4-2025-ts-ohlcv-sym", "source_version": 1, "source_license": "CC BY 4.0",
              "base_feature_mode": "expanded_regime", "target": target, "holdout_start": hold["timestamp"].iloc[0].isoformat(),
              "holdout_end": hold["timestamp"].iloc[-1].isoformat(), "holdout_base": _metric(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int)), "gates": {}}

    for mode, features in gate_modes.items():
        cols = [*features, "base_confidence", "base_pred"]
        # Joint correctness ranker, but discovery thresholds are learned separately for each predicted direction.
        joint = _gate_model().fit(meta[cols].to_numpy(float), meta["correct"].to_numpy(int))
        train_prob = joint.predict_proba(meta[cols].to_numpy(float))[:, 1]; hold_prob = joint.predict_proba(hold[cols].to_numpy(float))[:, 1]
        joint_thresholds = _thresholds(train_prob, meta["base_pred"].to_numpy(int))
        joint_row = {"auc": float(roc_auc_score(hold["correct"], hold_prob)), "brier": float(brier_score_loss(hold["correct"], hold_prob)),
                     "policy": _policy(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int), hold_prob, joint_thresholds),
                     "quarter_slices": _quarter_slices(hold, hold_prob, joint_thresholds)}

        # Separate correctness rankers per predicted direction remove residual direction-calibration coupling.
        train_sep = np.empty(len(meta), dtype=float); hold_sep = np.empty(len(hold), dtype=float); sep_thresholds = {}
        class_auc = {}
        for cls in (0, 1):
            mt = meta["base_pred"].to_numpy(int) == cls; mh = hold["base_pred"].to_numpy(int) == cls
            gate = _gate_model().fit(meta.loc[mt, cols].to_numpy(float), meta.loc[mt, "correct"].to_numpy(int))
            train_sep[mt] = gate.predict_proba(meta.loc[mt, cols].to_numpy(float))[:, 1]
            hold_sep[mh] = gate.predict_proba(hold.loc[mh, cols].to_numpy(float))[:, 1]
            values = train_sep[mt]; sep_thresholds[cls] = {"low": float(np.quantile(values, .25)), "high": float(np.quantile(values, .65))}
            class_auc[str(cls)] = float(roc_auc_score(hold.loc[mh, "correct"], hold_sep[mh]))
        separate_row = {"class_auc": class_auc, "auc": float(roc_auc_score(hold["correct"], hold_sep)),
                        "policy": _policy(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int), hold_sep, sep_thresholds),
                        "quarter_slices": _quarter_slices(hold, hold_sep, sep_thresholds)}
        result["gates"][mode] = {"joint_class_thresholds": joint_row, "separate_direction_gates": separate_row}
        print(mode, "joint", joint_row["policy"].get("selected", {}).get("balanced_accuracy"), "separate", separate_row["policy"].get("selected", {}).get("balanced_accuracy"))

    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(); result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("LICENSED_MNQ_CLASS_BALANCED_GATE=PASS"); print("RECEIPT_SHA256=" + result["receipt_sha256"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
