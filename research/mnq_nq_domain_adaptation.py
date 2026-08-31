from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_opportunity_target_matrix import classification, economic, model, target_columns
from research.mnq_to_nq_transfer import bars12, load_nq

CONFIGS = {
    "h12_vol10": (12, 1.0),
    "h24_vol05": (24, 0.5),
}
MNQ_CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
NQ_HOLDOUT_START = pd.Timestamp("2026-03-16", tz="UTC")


def fit_equal_market(features: list[str], mnq: pd.DataFrame, nq: pd.DataFrame):
    combined = pd.concat([mnq, nq], ignore_index=True)
    y = combined["target"].astype(int).to_numpy()
    weights = np.concatenate([
        np.full(len(mnq), 0.5 / len(mnq), dtype=float),
        np.full(len(nq), 0.5 / len(nq), dtype=float),
    ])
    weights *= len(combined)
    fitted = model()
    fitted.fit(
        combined[features].to_numpy(float),
        y,
        scale__sample_weight=weights,
        model__sample_weight=weights,
    )
    return fitted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--nq-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, mult = CONFIGS[args.config_key]

    mnq_raw = load_deep(args.deep_root)
    mnq = _add_features(deep_bars(stitch_deep(mnq_raw, deep_roll_schedule(mnq_raw))))
    nq = _add_features(bars12(load_nq(args.nq_csv)))

    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    feature_sets = {"baseline20": list(BASE_FEATURES), "expanded_regime": expanded}
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *expanded]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    mnq_label, _, _ = target_columns(mnq, horizon, mult)
    nq_label, nq_fwd, _ = target_columns(nq, horizon, mult)
    mnq["target"] = mnq_label
    nq["target"] = nq_label
    nq["fwd"] = nq_fwd

    mnq_train = mnq[(mnq["timestamp"] < MNQ_CUTOFF) & mnq["target"].notna()].copy()
    nq_train = nq[(nq["timestamp"] < NQ_HOLDOUT_START) & nq["target"].notna()].copy()
    nq_test = nq[(nq["timestamp"] >= NQ_HOLDOUT_START) & nq["target"].notna()].copy()
    if len(mnq_train) < 50000 or len(nq_train) < 2500 or len(nq_test) < 1000:
        raise RuntimeError(f"insufficient rows mnq={len(mnq_train)} nq_train={len(nq_train)} nq_test={len(nq_test)}")
    if nq_train["timestamp"].max() >= nq_test["timestamp"].min():
        raise RuntimeError("NQ adaptation chronology overlap")

    y_test = nq_test["target"].astype(int).to_numpy()
    fwd = nq_test["fwd"].to_numpy(float)
    result_sets = {}
    for fname, features in feature_sets.items():
        arms = {}
        mnq_model = model().fit(mnq_train[features].to_numpy(float), mnq_train["target"].astype(int).to_numpy())
        nq_model = model().fit(nq_train[features].to_numpy(float), nq_train["target"].astype(int).to_numpy())
        pooled_model = fit_equal_market(features, mnq_train, nq_train)
        for arm, fitted in {
            "mnq_frozen": mnq_model,
            "nq_recent_specialist": nq_model,
            "equal_market_pooled": pooled_model,
        }.items():
            pred = fitted.predict(nq_test[features].to_numpy(float)).astype(int)
            arms[arm] = {
                "classification": classification(y_test, pred),
                "economic": economic(pred, fwd),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
            }
            print(args.config_key, fname, arm, "BA", arms[arm]["classification"]["balanced_accuracy"], "NET2", None if arms[arm]["economic"] is None else arms[arm]["economic"]["net_mean_after_2bp"])
        result_sets[fname] = arms

    result = {
        "schema": "foundry.mnq_nq_domain_adaptation.v1",
        "research_only": True,
        "promotion_authority": False,
        "mnq_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "nq_source": "axb0306/cme-futures-ohlc@60abd3fb6369c6ce0b6a4a65b0f2562fc96b1264:NQ/NQ_1min_20260120_20260415.csv",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": mult,
        "protocol": "fixed NQ holdout beginning 2026-03-16; compare MNQ-only frozen transfer, NQ-only recent specialist, and pooled model with equal total MNQ/NQ sample weight; identical holdout timestamps for every arm; no holdout calibration or model selection",
        "mnq_train_rows": int(len(mnq_train)),
        "nq_train_rows": int(len(nq_train)),
        "nq_test_rows": int(len(nq_test)),
        "nq_train_last_timestamp": nq_train["timestamp"].max().isoformat(),
        "nq_test_first_timestamp": nq_test["timestamp"].min().isoformat(),
        "nq_test_last_timestamp": nq_test["timestamp"].max().isoformat(),
        "excluded_forward_aligned_features": ["chikou_span"],
        "feature_sets": result_sets,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_NQ_DOMAIN_ADAPTATION=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
