from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.licensed_mnq_expanded_validation import _bars, _load_aggregate, _roll_schedule, _stitch

HORIZON = 12
BASE_MODE = "expanded_regime"

LOCAL_STATE = [
    "vol_ratio_12_120", "atr14_pct", "atr28_pct", "rv_12", "rv_60", "rv_120",
    "z_volume_120", "z_close_20", "z_close_60", "range_z_60",
    "ema20_50_spread", "ema50_200_spread", "trend_slope_20", "trend_slope_60",
    "efficiency_10", "efficiency_30", "ret_skew_60", "ret_autocorr_60",
    "bb20_width", "bb20_pos", "donchian55_pos", "mfi_14", "rsi_14", "vwap_dist_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def _folds(n: int, horizon: int) -> list[tuple[int, int, int, int]]:
    first = n // 2
    size = (n - first) // 4
    if size < 500:
        raise RuntimeError(f"insufficient fold size: {size}")
    out = []
    for i in range(4):
        test_start = first + i * size
        test_end = n if i == 3 else first + (i + 1) * size
        train_end = test_start - horizon
        if train_end <= 2000:
            raise RuntimeError("insufficient purged training rows")
        out.append((0, train_end, test_start, test_end))
    return out


def _base_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42, C=0.5)),
    ])


def _gate_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=43, C=0.25)),
    ])


def _policy(y: np.ndarray, base: np.ndarray, probability: np.ndarray, train_probability: np.ndarray) -> dict:
    low = float(np.quantile(train_probability, 0.25))
    high = float(np.quantile(train_probability, 0.65))
    invert = probability <= low
    trust = probability >= high
    selected = invert | trust
    out = {
        "low_cut": low,
        "high_cut": high,
        "coverage": float(selected.mean()),
        "trust_coverage": float(trust.mean()),
        "invert_coverage": float(invert.mean()),
        "trust_rows": int(trust.sum()),
        "invert_rows": int(invert.sum()),
    }
    if trust.sum() >= 100:
        out["trust"] = _metric(y[trust], base[trust])
    if invert.sum() >= 100:
        out["invert"] = _metric(y[invert], 1 - base[invert])
    if selected.sum() >= 200:
        pred = base.copy()
        pred[invert] = 1 - pred[invert]
        out["selected"] = _metric(y[selected], pred[selected])
    return out


def _time_slices(hold: pd.DataFrame, probability: np.ndarray, train_probability: np.ndarray) -> dict:
    low = float(np.quantile(train_probability, 0.25))
    high = float(np.quantile(train_probability, 0.65))
    node = hold[["timestamp", "truth", "base_pred"]].copy()
    node["gate_probability"] = probability
    node["period"] = node["timestamp"].dt.to_period("Q").astype(str)
    receipts = {}
    for period, part in node.groupby("period", sort=True):
        if len(part) < 250:
            continue
        y = part["truth"].to_numpy(int)
        base = part["base_pred"].to_numpy(int)
        prob = part["gate_probability"].to_numpy(float)
        trust = prob >= high
        invert = prob <= low
        selected = trust | invert
        row = {"rows": int(len(part)), "base": _metric(y, base), "selected_coverage": float(selected.mean())}
        if selected.sum() >= 100:
            pred = base.copy(); pred[invert] = 1 - pred[invert]
            row["selected"] = _metric(y[selected], pred[selected])
        receipts[period] = row
    return receipts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = _load_aggregate(args.aggregate)
    stitched = _stitch(raw, _roll_schedule(raw))
    bars = _bars(stitched)
    frame = _add_features(bars)

    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES))
    base_features = list(dict.fromkeys(expanded + REGIME_FEATURES))
    target = "target_dir_h12"
    gate_modes = {
        "confidence_only": [],
        "regime_only": list(REGIME_FEATURES),
        "local_state": list(dict.fromkeys(REGIME_FEATURES + LOCAL_STATE)),
        "full_state": base_features,
    }
    all_gate = list(dict.fromkeys(base_features + LOCAL_STATE))
    cols = list(dict.fromkeys(["timestamp", *base_features, *all_gate, target]))
    work = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    folds = _folds(len(work), HORIZON)
    X = work[base_features].to_numpy(float)
    y = work[target].astype(int).to_numpy()

    meta_parts = []
    discovery_base = []
    for fold, (start, train_end, test_start, test_end) in enumerate(folds[:3]):
        model = _base_model().fit(X[start:train_end], y[start:train_end])
        pred = model.predict(X[test_start:test_end]).astype(int)
        prob_up = model.predict_proba(X[test_start:test_end])[:, 1]
        part = work.iloc[test_start:test_end][["timestamp", *all_gate]].copy()
        part["base_pred"] = pred
        part["base_confidence"] = np.abs(prob_up - 0.5) * 2.0
        part["correct"] = (pred == y[test_start:test_end]).astype(int)
        part["truth"] = y[test_start:test_end]
        part["fold"] = fold
        meta_parts.append(part)
        discovery_base.append({"fold": fold, **_metric(y[test_start:test_end], pred)})
    meta = pd.concat(meta_parts, ignore_index=True)

    _, train_end, test_start, test_end = folds[3]
    final_base = _base_model().fit(X[:train_end], y[:train_end])
    hold_pred = final_base.predict(X[test_start:test_end]).astype(int)
    hold_up = final_base.predict_proba(X[test_start:test_end])[:, 1]
    hold = work.iloc[test_start:test_end][["timestamp", *all_gate]].copy()
    hold["base_pred"] = hold_pred
    hold["base_confidence"] = np.abs(hold_up - 0.5) * 2.0
    hold["correct"] = (hold_pred == y[test_start:test_end]).astype(int)
    hold["truth"] = y[test_start:test_end]

    result = {
        "schema": "foundry.licensed_mnq_trust_gate.v1",
        "research_only": True,
        "promotion_authority": False,
        "source_dataset": "brandenmorris/mnq-1m-q4-2022-q4-2025-ts-ohlcv-sym",
        "source_version": 1,
        "source_license": "CC BY 4.0",
        "base_feature_mode": BASE_MODE,
        "base_feature_count": len(base_features),
        "target": target,
        "excluded_forward_aligned_features": ["chikou_span"],
        "rows": int(len(work)),
        "meta_rows": int(len(meta)),
        "holdout_rows": int(len(hold)),
        "holdout_start": hold["timestamp"].iloc[0].isoformat(),
        "holdout_end": hold["timestamp"].iloc[-1].isoformat(),
        "discovery_base_folds": discovery_base,
        "holdout_base": _metric(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int)),
        "gates": {},
    }

    for mode, features in gate_modes.items():
        gate_cols = [*features, "base_confidence", "base_pred"]
        gate = _gate_model().fit(meta[gate_cols].to_numpy(float), meta["correct"].to_numpy(int))
        train_prob = gate.predict_proba(meta[gate_cols].to_numpy(float))[:, 1]
        hold_prob = gate.predict_proba(hold[gate_cols].to_numpy(float))[:, 1]
        correct = hold["correct"].to_numpy(int)
        row = {
            "feature_count": len(gate_cols),
            "correctness_auc": float(roc_auc_score(correct, hold_prob)),
            "brier": float(brier_score_loss(correct, hold_prob)),
            "policy": _policy(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int), hold_prob, train_prob),
            "quarter_slices": _time_slices(hold, hold_prob, train_prob),
        }
        result["gates"][mode] = row
        selected = row["policy"].get("selected", {}).get("balanced_accuracy")
        print(f"GATE={mode} AUC={row['correctness_auc']:.6f} SELECTED_BA={selected}")

    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LICENSED_MNQ_TRUST_GATE=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
