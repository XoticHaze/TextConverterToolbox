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


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--source-root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    frames = {}
    valid_ts = {}
    for root in ab.ROOTS:
        matches = glob.glob(str(args.source_root / root / f"{root}_1min_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"{root}: source mismatch {matches}")
        f = ab._add_features(ab._build_bars(Path(matches[0]), root))
        frames[root] = f
        req = f[["timestamp", *ab.BASE_FEATURES]].replace([np.inf, -np.inf], np.nan).dropna()
        valid_ts[root] = set(req["timestamp"].tolist())
    aligned = set.intersection(*(valid_ts[r] for r in ab.ROOTS))
    if not aligned:
        raise RuntimeError("no fully aligned timestamps")

    result = {
        "schema": "foundry.crossmarket_alignment_audit.v1",
        "research_only": True,
        "promotion_authority": False,
        "source_commit": ab.SOURCE_COMMIT,
        "target": "target_dir_h12",
        "alignment_definition": "same timestamp has complete baseline20 causal features for CL/GC/MNQ/NQ/ZN",
        "aligned_timestamp_count": len(aligned),
        "markets": {},
    }

    for root, frame in frames.items():
        work = frame[["timestamp", *ab.BASE_FEATURES, "target_dir_h12"]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        folds = ab._row_folds(len(work), 12)
        _, train_end, test_start, test_end = folds[3]
        X = work[ab.BASE_FEATURES].to_numpy(float); y = work["target_dir_h12"].astype(int).to_numpy()
        m = model().fit(X[:train_end], y[:train_end]); pred = m.predict(X[test_start:test_end]).astype(int)
        hold = work.iloc[test_start:test_end][["timestamp"]].copy(); hold["truth"] = y[test_start:test_end]; hold["pred"] = pred
        mask = hold["timestamp"].isin(aligned).to_numpy()
        if mask.sum() < 100 or (~mask).sum() < 100:
            raise RuntimeError(f"{root}: insufficient aligned/nonaligned holdout rows {mask.sum()}/{(~mask).sum()}")
        hour = hold["timestamp"].dt.hour
        aligned_hours = hour[mask].value_counts(normalize=True).sort_index()
        nonaligned_hours = hour[~mask].value_counts(normalize=True).sort_index()
        result["markets"][root] = {
            "holdout_rows": int(len(hold)),
            "aligned_rows": int(mask.sum()),
            "aligned_coverage": float(mask.mean()),
            "full": metric(hold["truth"].to_numpy(int), hold["pred"].to_numpy(int)),
            "aligned": metric(hold.loc[mask, "truth"].to_numpy(int), hold.loc[mask, "pred"].to_numpy(int)),
            "nonaligned": metric(hold.loc[~mask, "truth"].to_numpy(int), hold.loc[~mask, "pred"].to_numpy(int)),
            "aligned_hour_distribution_utc": {str(int(k)): float(v) for k, v in aligned_hours.items()},
            "nonaligned_hour_distribution_utc": {str(int(k)): float(v) for k, v in nonaligned_hours.items()},
            "aligned_first": hold.loc[mask, "timestamp"].iloc[0].isoformat(),
            "aligned_last": hold.loc[mask, "timestamp"].iloc[-1].isoformat(),
        }
        node = result["markets"][root]
        print(root, "FULL", node["full"]["balanced_accuracy"], "ALIGNED", node["aligned"]["balanced_accuracy"], "NONALIGNED", node["nonaligned"]["balanced_accuracy"], "COVERAGE", node["aligned_coverage"])

    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(); result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CROSSMARKET_ALIGNMENT_AUDIT=PASS"); print("RECEIPT_SHA256=" + result["receipt_sha256"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
