from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research import expanded_regime_ablation as ab
from research.h12_relational_trust_gate import LOCAL_STATE, add_relational, gate_model, metric, policy, prepare


def _neutralize_structural_missing(frame: pd.DataFrame, relational_union: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in relational_union:
        if col not in out.columns:
            out[col] = 0.0
    out[relational_union] = out[relational_union].fillna(0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--source-root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    frames = {}
    for root in ab.ROOTS:
        matches = glob.glob(str(args.source_root / root / f"{root}_1min_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"{root}: source mismatch {matches}")
        frames[root] = ab._add_features(ab._build_bars(Path(matches[0]), root))
    frames, relational_union = add_relational(frames)

    prepared = {}; pooled_meta = []; pooled_hold = []
    for root, frame in frames.items():
        meta, hold, rel = prepare(root, frame)
        prepared[root] = (meta, hold, rel)
        pooled_meta.append(_neutralize_structural_missing(meta, relational_union))
        pooled_hold.append(_neutralize_structural_missing(hold, relational_union))

    result = {
        "schema": "foundry.h12_relational_trust_gate.v2",
        "research_only": True,
        "promotion_authority": False,
        "source_commit": ab.SOURCE_COMMIT,
        "target": "target_dir_h12",
        "relational_feature_union_count": len(relational_union),
        "pooled_structural_missing_policy": "self/not-applicable relational columns are neutral-zero encoded with market IDs",
        "markets": {},
        "pooled": {},
    }

    for root, (meta, hold, rel) in prepared.items():
        modes = {
            "regime_only": list(ab.REGIME_FEATURES),
            "relational_only": rel,
            "state_relational": list(dict.fromkeys([*ab.REGIME_FEATURES, *LOCAL_STATE, *rel])),
        }
        result["markets"][root] = {}
        for name, features in modes.items():
            cols = [*features, "base_confidence", "base_pred"]
            gate = gate_model().fit(meta[cols].to_numpy(float), meta["correct"].to_numpy(int))
            train_prob = gate.predict_proba(meta[cols].to_numpy(float))[:, 1]
            hold_prob = gate.predict_proba(hold[cols].to_numpy(float))[:, 1]
            result["markets"][root][name] = {
                "feature_count": len(cols),
                "auc": float(roc_auc_score(hold["correct"], hold_prob)),
                "base": metric(hold["truth"], hold["base_pred"]),
                "policy": policy(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int), hold_prob, train_prob),
            }
        print(root, {k: round(v["policy"].get("selected", {}).get("balanced_accuracy", 0), 4) for k, v in result["markets"][root].items()})

    meta = pd.concat(pooled_meta, ignore_index=True); hold = pd.concat(pooled_hold, ignore_index=True)
    ids = []
    for root in ab.ROOTS:
        col = f"market_{root}"; ids.append(col)
        meta[col] = (meta["market"] == root).astype(float); hold[col] = (hold["market"] == root).astype(float)
    pooled_modes = {
        "regime_only": list(ab.REGIME_FEATURES),
        "relational_only": relational_union,
        "state_relational": list(dict.fromkeys([*ab.REGIME_FEATURES, *LOCAL_STATE, *relational_union])),
    }
    for name, features in pooled_modes.items():
        cols = [*features, *ids, "base_confidence", "base_pred"]
        if meta[cols].isna().any().any() or hold[cols].isna().any().any():
            raise RuntimeError(f"pooled relational encoding still contains NaN for {name}")
        gate = gate_model().fit(meta[cols].to_numpy(float), meta["correct"].to_numpy(int))
        train_prob = gate.predict_proba(meta[cols].to_numpy(float))[:, 1]
        hold_prob = gate.predict_proba(hold[cols].to_numpy(float))[:, 1]
        row = {
            "feature_count": len(cols),
            "auc": float(roc_auc_score(hold["correct"], hold_prob)),
            "base": metric(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int)),
            "policy": policy(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int), hold_prob, train_prob),
            "per_market": {},
        }
        for root in ab.ROOTS:
            mask = hold["market"].to_numpy() == root
            row["per_market"][root] = policy(
                hold.loc[mask, "truth"].to_numpy(int), hold.loc[mask, "base_pred"].to_numpy(int), hold_prob[mask], train_prob
            )
        result["pooled"][name] = row
        print("POOLED", name, round(row["policy"].get("selected", {}).get("balanced_accuracy", 0), 4))

    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(); result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("H12_RELATIONAL_TRUST_GATE_V2=PASS"); print("RECEIPT_SHA256=" + result["receipt_sha256"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
