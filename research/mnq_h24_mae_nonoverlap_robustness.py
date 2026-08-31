from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.expanded_regime_ablation import BASE_FEATURES
from research.mnq_expected_move_axb_2026 import AXB_PIN, CUTOFF, HORIZON, load_axb_mnq
from research.mnq_external_transfer_validation import deep_bars, deep_roll_schedule, stitch_deep
from research.mnq_h24_mae_axb_transfer import _corr, _prepare, _spearman
from research.mnq_h24_mae_risk_specialist import risk_model
from research.deep_mnq_source_contract import contract_receipt, load_deep

MIN_TEST_ROWS = 3000
MIN_PHASE_ROWS = 200


def _phase_indices(n_rows: int, phase: int, horizon: int = HORIZON) -> np.ndarray:
    if not 0 <= phase < horizon:
        raise ValueError(f"phase must be in [0,{horizon}), got {phase}")
    return np.arange(phase, n_rows, horizon, dtype=int)


def _phase_receipts(test, pred: np.ndarray, target: str) -> list[dict]:
    realized = test[target].to_numpy(float)
    rows = []
    for phase in range(HORIZON):
        idx = _phase_indices(len(test), phase)
        if len(idx) < MIN_PHASE_ROWS:
            raise RuntimeError(f"phase {phase} has only {len(idx)} rows")
        if len(idx) > 1 and int(np.diff(idx).min()) < HORIZON:
            raise RuntimeError(f"phase {phase} is not H{HORIZON} non-overlapping")
        rows.append(
            {
                "phase": phase,
                "rows": int(len(idx)),
                "first_timestamp": test.iloc[int(idx[0])]["timestamp"].isoformat(),
                "last_timestamp": test.iloc[int(idx[-1])]["timestamp"].isoformat(),
                "pearson": _corr(pred[idx], realized[idx]),
                "spearman": _spearman(pred[idx], realized[idx]),
            }
        )
    return rows


def _summarize(phases: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in phases if row[key] is not None], dtype=float)
    if len(values) != HORIZON:
        raise RuntimeError(f"expected {HORIZON} finite {key} phases, got {len(values)}")
    return {
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "positive_phases": int(np.sum(values > 0)),
        "phase_count": int(len(values)),
    }


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
        model = risk_model()
        model.fit(x_train, train[target].to_numpy(float))
        pred = np.asarray(model.predict(x_test), dtype=float)
        realized = test[target].to_numpy(float)
        phases = _phase_receipts(test, pred, target)
        sides[side] = {
            "full_overlap_reference": {
                "rows": int(len(test)),
                "pearson": _corr(pred, realized),
                "spearman": _spearman(pred, realized),
            },
            "nonoverlap_stride_bars": HORIZON,
            "phases": phases,
            "pearson_phase_summary": _summarize(phases, "pearson"),
            "spearman_phase_summary": _summarize(phases, "spearman"),
        }
        print(side, json.dumps(sides[side], sort_keys=True))

    result = {
        "schema": "foundry.mnq_h24_mae_nonoverlap_robustness.v1",
        "research_only": True,
        "promotion_authority": False,
        "policy_authority": False,
        "training_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176 strictly pre-2026",
        "test_source": f"axb0306/cme-futures-ohlc@{AXB_PIN}",
        "deep_timestamp_contract": contract_receipt(),
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "risk_model": "same frozen HistGradientBoostingRegressor quantile=0.80 family as corrected H24 MAE transfer; separate long/short models",
        "contract": "robustness-only reuse of already-consumed AXB Jan-Apr 2026; fit unchanged pre-2026 models; report all 24 stride-H24 phase correlations so each phase contains non-overlapping future windows; no phase selection, thresholding, PnL, sizing, calibration, or promotion inference",
        "train_rows": int(len(train)),
        "train_last_timestamp": train["timestamp"].max().isoformat(),
        "test_rows": int(len(test)),
        "test_first_timestamp": test["timestamp"].min().isoformat(),
        "test_last_timestamp": test["timestamp"].max().isoformat(),
        "phase_count": HORIZON,
        "sides": sides,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_MAE_NONOVERLAP_ROBUSTNESS=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
