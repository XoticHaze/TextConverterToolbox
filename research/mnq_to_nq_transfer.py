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

CONFIGS = {
    "h12_vol10": (12, 1.0),
    "h24_vol05": (24, 0.5),
}
TRAIN_CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")


def load_nq(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    expected = ["datetime", "open", "high", "low", "close", "volume"]
    if list(f.columns) != expected:
        raise RuntimeError(f"unexpected NQ schema {list(f.columns)}")
    f["timestamp"] = pd.to_datetime(f["datetime"], utc=True, errors="raise")
    for c in ["open", "high", "low", "close", "volume"]:
        f[c] = pd.to_numeric(f[c], errors="raise")
    f = f.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
    if f["timestamp"].duplicated().any() or not f["timestamp"].is_monotonic_increasing:
        raise RuntimeError("NQ timestamps not unique/increasing")
    return f[["timestamp", "open", "high", "low", "close", "volume"]]


def bars12(frame: pd.DataFrame) -> pd.DataFrame:
    w = frame.set_index("timestamp")
    b = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), observed_minutes=("close", "count"),
    )
    b = b[b["observed_minutes"] > 0].reset_index()
    b["market"] = "NQ"
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--nq-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, mult = CONFIGS[args.config_key]

    mnq_raw = load_deep(args.deep_root)
    mnq_stitched = stitch_deep(mnq_raw, deep_roll_schedule(mnq_raw))
    mnq = _add_features(deep_bars(mnq_stitched))
    nq_raw = load_nq(args.nq_csv)
    nq = _add_features(bars12(nq_raw))

    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    feature_sets = {"baseline20": list(BASE_FEATURES), "expanded_regime": expanded}
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *expanded]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    mnq_label, _, _ = target_columns(mnq, horizon, mult)
    nq_label, nq_fwd, nq_threshold = target_columns(nq, horizon, mult)
    mnq["target"] = mnq_label
    nq["target"] = nq_label
    nq["fwd"] = nq_fwd
    nq["threshold"] = nq_threshold

    train = mnq[(mnq["timestamp"] < TRAIN_CUTOFF) & mnq["target"].notna()].copy()
    test = nq[nq["target"].notna()].copy()
    if len(train) < 50000:
        raise RuntimeError(f"insufficient MNQ train rows {len(train)}")
    if len(test) < 3000:
        raise RuntimeError(f"insufficient NQ test rows {len(test)}")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("transfer chronology overlap: MNQ training reaches NQ test period")

    result_sets = {}
    y_train = train["target"].astype(int).to_numpy()
    y_test = test["target"].astype(int).to_numpy()
    fwd = test["fwd"].to_numpy(float)
    for name, features in feature_sets.items():
        fitted = model().fit(train[features].to_numpy(float), y_train)
        pred = fitted.predict(test[features].to_numpy(float)).astype(int)
        result_sets[name] = {
            "classification": classification(y_test, pred),
            "economic": economic(pred, fwd),
            "train_target_counts": {str(c): int((y_train == c).sum()) for c in (-1, 0, 1)},
            "test_target_counts": {str(c): int((y_test == c).sum()) for c in (-1, 0, 1)},
            "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
        }
        print(args.config_key, name, "BA", result_sets[name]["classification"]["balanced_accuracy"], "NET2", None if result_sets[name]["economic"] is None else result_sets[name]["economic"]["net_mean_after_2bp"])

    result = {
        "schema": "foundry.mnq_to_nq_transfer.v1",
        "research_only": True,
        "promotion_authority": False,
        "train_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "test_source": "axb0306/cme-futures-ohlc@60abd3fb6369c6ce0b6a4a65b0f2562fc96b1264:NQ/NQ_1min_20260120_20260415.csv",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": mult,
        "protocol": "fit once on MNQ rows strictly before 2026-01-01; freeze scaler/model; apply unchanged to NQ 2026; no NQ labels or outcomes used for fitting/calibration/feature selection",
        "train_rows": int(len(train)),
        "train_last_timestamp": train["timestamp"].max().isoformat(),
        "test_rows": int(len(test)),
        "test_first_timestamp": test["timestamp"].min().isoformat(),
        "test_last_timestamp": test["timestamp"].max().isoformat(),
        "median_nq_causal_threshold": float(np.median(test["threshold"].to_numpy(float))),
        "excluded_forward_aligned_features": ["chikou_span"],
        "feature_sets": result_sets,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_TO_NQ_TRANSFER=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
