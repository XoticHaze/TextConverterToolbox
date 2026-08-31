from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_opportunity_target_matrix import COST_FLOOR, classification, economic, model, quarter_starts, target_columns

CONFIGS = {
    "h6_vol05": (6, 0.5),
    "h6_vol10": (6, 1.0),
    "h12_vol05": (12, 0.5),
    "h12_vol10": (12, 1.0),
    "h24_vol05": (24, 0.5),
    "h24_vol10": (24, 1.0),
}
BAR_NS = 12 * 60 * 1_000_000_000


def phases(horizon: int) -> list[int]:
    return sorted(set([0, horizon // 4, horizon // 2, (3 * horizon) // 4]))


def utc_slot(ts: pd.Series) -> np.ndarray:
    # Normalize explicitly: pandas may preserve microsecond-resolution input,
    # while BAR_NS is expressed in nanoseconds. Integer division is valid only
    # after both operands use the same unit.
    ns = pd.to_datetime(ts, utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return (ns // BAR_NS).astype(np.int64)


def phase_summary(phase_rows: dict[str, dict]) -> dict:
    net2 = [v["aggregate_economic"]["net_mean_after_2bp"] for v in phase_rows.values() if v["aggregate_economic"]]
    gross = [v["aggregate_economic"]["gross_mean_forward_return"] for v in phase_rows.values() if v["aggregate_economic"]]
    return {
        "phases": len(phase_rows),
        "phases_positive_after_2bp": int(sum(v > 0 for v in net2)),
        "median_phase_net_after_2bp": float(np.median(net2)) if net2 else None,
        "min_phase_net_after_2bp": float(np.min(net2)) if net2 else None,
        "max_phase_net_after_2bp": float(np.max(net2)) if net2 else None,
        "median_phase_gross": float(np.median(gross)) if gross else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    horizon, mult = CONFIGS[args.config_key]
    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)

    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    feature_sets = {"baseline20": list(BASE_FEATURES), "expanded_regime": expanded}
    common_cols = list(dict.fromkeys(["timestamp", "close", "rv_120", *expanded]))
    work = frame[common_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, fwd, threshold = target_columns(work, horizon, mult)
    work["target"] = label
    work["fwd"] = fwd
    work["threshold"] = threshold
    work["utc_slot"] = utc_slot(work["timestamp"])

    outer = quarter_starts()
    phase_ids = phases(horizon)
    result_features = {}

    for feature_name, features in feature_sets.items():
        by_phase = {str(p): {"truth": [], "pred": [], "fwd": [], "quarter_rows": []} for p in phase_ids}
        for i in range(len(outer) - 1):
            start, end = outer[i], outer[i + 1]
            test_mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna()
            test_idx = np.flatnonzero(test_mask.to_numpy())
            if len(test_idx) < 500:
                continue
            test_start = int(test_idx[0]); test_end = int(test_idx[-1] + 1)
            train_end = test_start - horizon
            if train_end < 10000:
                continue
            train = work.iloc[:train_end]
            train = train[train["target"].notna()]
            test = work.iloc[test_start:test_end]
            test = test[test["target"].notna()].copy()
            if len(train) < 10000 or len(test) < 500:
                continue
            y_train = train["target"].astype(int).to_numpy()
            y_test = test["target"].astype(int).to_numpy()
            if len(np.unique(y_train)) < 3 or len(np.unique(y_test)) < 3:
                continue
            m = model().fit(train[features].to_numpy(float), y_train)
            pred = m.predict(test[features].to_numpy(float)).astype(int)
            fwd_test = test["fwd"].to_numpy(float)
            slots = test["utc_slot"].to_numpy(np.int64)

            for p in phase_ids:
                pmask = (slots % horizon) == p
                if pmask.sum() < 50:
                    continue
                y_p = y_test[pmask]; pred_p = pred[pmask]; fwd_p = fwd_test[pmask]
                econ = economic(pred_p, fwd_p)
                node = by_phase[str(p)]
                node["truth"].extend(y_p.tolist()); node["pred"].extend(pred_p.tolist()); node["fwd"].extend(fwd_p.tolist())
                node["quarter_rows"].append({
                    "period": f"{start.year}Q{((start.month - 1)//3)+1}",
                    "rows": int(pmask.sum()),
                    "classification": classification(y_p, pred_p),
                    "economic": econ,
                })

        phase_results = {}
        for p, node in by_phase.items():
            truth = np.asarray(node.pop("truth"), dtype=int)
            pred = np.asarray(node.pop("pred"), dtype=int)
            fwd_arr = np.asarray(node.pop("fwd"), dtype=float)
            if len(truth) < 500:
                raise RuntimeError(f"{args.config_key}/{feature_name}/phase{p}: insufficient aggregate rows {len(truth)}")
            q_econ = [r["economic"]["net_mean_after_2bp"] for r in node["quarter_rows"] if r["economic"]]
            phase_results[p] = {
                **node,
                "aggregate_rows": int(len(truth)),
                "aggregate_classification": classification(truth, pred),
                "aggregate_economic": economic(pred, fwd_arr),
                "quarters_positive_after_2bp": int(sum(v > 0 for v in q_econ)),
                "median_quarter_net_after_2bp": float(np.median(q_econ)) if q_econ else None,
            }
            e = phase_results[p]["aggregate_economic"]
            print(args.config_key, feature_name, "PHASE", p, "ROWS", len(truth), "BA", phase_results[p]["aggregate_classification"]["balanced_accuracy"], "NET2", None if e is None else e["net_mean_after_2bp"])

        result_features[feature_name] = {
            "phases": phase_results,
            "phase_summary": phase_summary(phase_results),
        }

    result = {
        "schema": "foundry.mnq_nonoverlap_phase_audit.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": mult,
        "target": "down/no-trade/up using max(2bp, k*causal_rv120*sqrt(horizon))",
        "protocol": "same expanding past-only fitted models as opportunity matrix; economic/classification audit on four predeclared absolute-UTC phase streams separated by full horizon; phases fixed before results; no best-phase selection",
        "phase_offsets": phase_ids,
        "training_note": "training labels remain dense/overlapping; this audit isolates whether evaluation economics are an overlapping-outcome artifact before event-only retraining",
        "excluded_forward_aligned_features": ["chikou_span"],
        "feature_sets": result_features,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_NONOVERLAP_PHASE_AUDIT=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
