from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research import expanded_regime_ablation as ab
from research.h12_relational_trust_gate import LOCAL_STATE, add_relational


def metric(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
    }


def model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42, C=0.5)),
    ])


def safe_metric(frame: pd.DataFrame, mask: np.ndarray) -> dict | None:
    if int(mask.sum()) < 50:
        return None
    return metric(frame.loc[mask, "truth"].to_numpy(int), frame.loc[mask, "pred"].to_numpy(int))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--source-root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    raw_frames = {}
    baseline_valid = {}
    for root in ab.ROOTS:
        matches = glob.glob(str(args.source_root / root / f"{root}_1min_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"{root}: source mismatch {matches}")
        frame = ab._add_features(ab._build_bars(Path(matches[0]), root))
        raw_frames[root] = frame
        node = frame[["timestamp", *ab.BASE_FEATURES]].replace([np.inf, -np.inf], np.nan).dropna()
        baseline_valid[root] = set(node["timestamp"].tolist())

    aligned_baseline = set.intersection(*(baseline_valid[r] for r in ab.ROOTS))
    relational_frames, _ = add_relational(raw_frames)

    result = {
        "schema": "foundry.crossmarket_alignment_audit.v2",
        "research_only": True,
        "promotion_authority": False,
        "source_commit": ab.SOURCE_COMMIT,
        "target": "target_dir_h12",
        "question": "does cross-market completeness itself select an easier holdout subset before any gate is applied?",
        "markets": {},
    }

    for root in ab.ROOTS:
        base = raw_frames[root][["timestamp", *ab.BASE_FEATURES, "target_dir_h12"]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        folds = ab._row_folds(len(base), 12)
        _, train_end, test_start, test_end = folds[3]
        X = base[ab.BASE_FEATURES].to_numpy(float); y = base["target_dir_h12"].astype(int).to_numpy()
        m = model().fit(X[:train_end], y[:train_end]); pred = m.predict(X[test_start:test_end]).astype(int)
        hold = base.iloc[test_start:test_end][["timestamp"]].copy(); hold["truth"] = y[test_start:test_end]; hold["pred"] = pred

        rel = relational_frames[root]
        rel_features = rel.attrs["relational_features"]
        strict_features = list(dict.fromkeys([*ab.REGIME_FEATURES, *LOCAL_STATE, *rel_features]))
        strict = rel[["timestamp", *strict_features]].replace([np.inf, -np.inf], np.nan).dropna()
        strict_ts = set(strict["timestamp"].tolist())

        mask_baseline_align = hold["timestamp"].isin(aligned_baseline).to_numpy()
        mask_strict = hold["timestamp"].isin(strict_ts).to_numpy()
        masks = {
            "all_rows": np.ones(len(hold), dtype=bool),
            "baseline_crossmarket_complete": mask_baseline_align,
            "strict_relational_complete": mask_strict,
            "strict_relational_incomplete": ~mask_strict,
        }
        rows = {}
        for name, mask in masks.items():
            ts = hold.loc[mask, "timestamp"]
            rows[name] = {
                "rows": int(mask.sum()),
                "coverage": float(mask.mean()),
                "metric": safe_metric(hold, mask),
                "first": ts.iloc[0].isoformat() if len(ts) else None,
                "last": ts.iloc[-1].isoformat() if len(ts) else None,
                "hour_distribution_utc": {str(int(k)): float(v) for k, v in ts.dt.hour.value_counts(normalize=True).sort_index().items()} if len(ts) else {},
            }
        result["markets"][root] = rows
        print(root, {k: (v["rows"], None if v["metric"] is None else round(v["metric"]["balanced_accuracy"], 4)) for k, v in rows.items()})

    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(); result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CROSSMARKET_ALIGNMENT_AUDIT_V2=PASS"); print("RECEIPT_SHA256=" + result["receipt_sha256"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
