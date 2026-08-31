from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_expected_move_axb_2026 import AXB_PIN, CUTOFF, HORIZON, load_axb_mnq
from research.mnq_external_transfer_validation import deep_bars, deep_roll_schedule, stitch_deep
from research.mnq_h24_mae_risk_specialist import future_window_extreme, risk_model

MIN_TEST_ROWS = 3000


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    features = list(BASE_FEATURES)
    x = _add_features(frame)
    needed = list(dict.fromkeys(["timestamp", "open", "high", "low", "close", "rv_120", *features]))
    x = x[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    future_high = future_window_extreme(x["high"].astype(float), HORIZON, "max")
    future_low = future_window_extreme(x["low"].astype(float), HORIZON, "min")
    scale = x["close"].astype(float) * x["rv_120"].astype(float) * math.sqrt(HORIZON)
    x["long_mae_z"] = (x["close"] - future_low).clip(lower=0) / scale.replace(0, np.nan)
    x["short_mae_z"] = (future_high - x["close"]).clip(lower=0) / scale.replace(0, np.nan)
    return x


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 100 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 100:
        return None
    return float(a[mask].corr(b[mask], method="spearman"))


def _monthly(test: pd.DataFrame, pred: np.ndarray, target: str) -> list[dict]:
    t = test[["timestamp", target]].copy()
    t["pred"] = np.asarray(pred, dtype=float)
    t["month"] = pd.to_datetime(t["timestamp"], utc=True).dt.to_period("M").astype(str)
    rows = []
    for month, g in t.groupby("month", sort=True):
        if len(g) < 500:
            continue
        rows.append({
            "month": month,
            "rows": int(len(g)),
            "pearson": _corr(g["pred"].to_numpy(), g[target].to_numpy()),
            "spearman": _spearman(g["pred"].to_numpy(), g[target].to_numpy()),
            "predicted_mean": float(g["pred"].mean()),
            "realized_mean": float(g[target].mean()),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--axb-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    features = list(BASE_FEATURES)
    deep_raw = load_deep(args.deep_root)
    deep = _prepare(deep_bars(stitch_deep(deep_raw, deep_roll_schedule(deep_raw))))
    axb = _prepare(load_axb_mnq(args.axb_root))

    pre = deep[(deep["timestamp"] < CUTOFF) & deep["long_mae_z"].notna() & deep["short_mae_z"].notna()].copy()
    if len(pre) <= HORIZON:
        raise RuntimeError("insufficient pre-2026 deep rows")
    train = pre.iloc[:-HORIZON].copy()
    test = axb[(axb["timestamp"] >= CUTOFF) & axb["long_mae_z"].notna() & axb["short_mae_z"].notna()].copy()
    if len(train) < 50000 or len(test) < MIN_TEST_ROWS:
        raise RuntimeError(f"insufficient train/test rows train={len(train)} test={len(test)}")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("chronology overlap")

    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    sides = {}
    for side, target in (("long", "long_mae_z"), ("short", "short_mae_z")):
        m = risk_model()
        m.fit(x_train, train[target].to_numpy(float))
        pred = m.predict(x_test)
        realized = test[target].to_numpy(float)
        sides[side] = {
            "test_rows": int(len(test)),
            "pearson_predicted_vs_realized_mae_z": _corr(pred, realized),
            "spearman_predicted_vs_realized_mae_z": _spearman(pred, realized),
            "predicted_mae_z_mean": float(np.mean(pred)),
            "predicted_mae_z_median": float(np.median(pred)),
            "realized_mae_z_mean": float(np.mean(realized)),
            "realized_mae_z_median": float(np.median(realized)),
            "monthly": _monthly(test, pred, target),
        }
        print(side, json.dumps(sides[side], sort_keys=True))

    result = {
        "schema": "foundry.mnq_h24_mae_axb_transfer.v1",
        "research_only": True,
        "promotion_authority": False,
        "policy_authority": False,
        "training_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176 strictly pre-2026",
        "test_source": f"axb0306/cme-futures-ohlc@{AXB_PIN}",
        "deep_timestamp_contract": contract_receipt(),
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "risk_model": "same fixed HistGradientBoostingRegressor quantile=0.80 family as corrected H24 MAE research; separate long/short models",
        "contract": "fit once on corrected deep MNQ strictly pre-2026 with H24 purge; apply unchanged to independent AXB 2026; evaluate risk ranking only; no veto, PnL threshold, sizing, or test-set calibration",
        "train_rows": int(len(train)),
        "train_last_timestamp": train["timestamp"].max().isoformat(),
        "test_rows": int(len(test)),
        "test_first_timestamp": test["timestamp"].min().isoformat(),
        "test_last_timestamp": test["timestamp"].max().isoformat(),
        "sides": sides,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_MAE_AXB_TRANSFER=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
