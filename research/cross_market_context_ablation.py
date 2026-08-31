"""Chronological OOS ablation for causal cross-market context.

Research-only. Compares a target-market baseline against one or more causal
context feature sets on identical expanding chronological folds. This module
owns no runtime, trading, StrategySpec, corpus, or promotion authority.

Expected input is a pre-materialized target panel containing timestamp, close,
binary label, and causal target features. Context frames are raw timestamp/close
bars and are aligned exclusively through causal_cross_market_context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_cross_market_context import ContextSpec, assert_causal_context, build_context_features


def _fold_boundaries(n: int, min_train: int, test_size: int) -> list[tuple[int, int]]:
    if min_train <= 0 or test_size <= 0:
        raise ValueError("min_train and test_size must be positive")
    out = []
    start = min_train
    while start + test_size <= n:
        out.append((start, start + test_size))
        start += test_size
    if not out:
        raise ValueError("insufficient rows for one chronological fold")
    return out


def _score_panel(panel: pd.DataFrame, features: list[str], label: str, min_train: int, test_size: int, purge_rows: int) -> dict:
    rows = panel.dropna(subset=[label]).sort_values("timestamp").reset_index(drop=True)
    y = rows[label].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("label must be binary 0/1")
    fold_rows = []
    all_y: list[int] = []
    all_p: list[float] = []
    for train_end_raw, test_end in _fold_boundaries(len(rows), min_train, test_size):
        train_end = train_end_raw - purge_rows
        if train_end <= 0:
            raise ValueError("purge removes all training rows")
        tr = rows.iloc[:train_end]
        te = rows.iloc[train_end_raw:test_end]
        if tr[label].nunique() < 2 or te.empty:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(tr[features], tr[label].astype(int))
        p = model.predict_proba(te[features])[:, 1]
        pred = (p >= 0.5).astype(int)
        yt = te[label].astype(int).to_numpy()
        ba = float(balanced_accuracy_score(yt, pred))
        auc = float(roc_auc_score(yt, p)) if len(np.unique(yt)) == 2 else None
        fold_rows.append({"train_end": str(tr["timestamp"].iloc[-1]), "test_start": str(te["timestamp"].iloc[0]), "test_end": str(te["timestamp"].iloc[-1]), "n": int(len(te)), "balanced_accuracy": ba, "roc_auc": auc})
        all_y.extend(yt.tolist()); all_p.extend(p.tolist())
    if not fold_rows:
        raise ValueError("no valid chronological folds")
    ya = np.asarray(all_y); pa = np.asarray(all_p)
    return {"n_oos": int(len(ya)), "balanced_accuracy": float(balanced_accuracy_score(ya, (pa >= 0.5).astype(int))), "roc_auc": float(roc_auc_score(ya, pa)) if len(np.unique(ya)) == 2 else None, "folds": fold_rows}


def run_ablation(target: pd.DataFrame, contexts: Mapping[str, pd.DataFrame], *, label: str, target_features: list[str], min_train: int, test_size: int, purge_rows: int, max_staleness: str = "15min") -> dict:
    required = {"timestamp", "close", label, *target_features}
    missing = required - set(target.columns)
    if missing:
        raise ValueError(f"target missing columns {sorted(missing)}")
    if label in target_features:
        raise ValueError("label cannot be a feature")
    if purge_rows < 0:
        raise ValueError("purge_rows must be non-negative")
    target = target.copy()
    target["timestamp"] = pd.to_datetime(target["timestamp"], utc=True, errors="raise")
    target = target.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ctx = build_context_features(target[["timestamp", "close"]], contexts, ContextSpec(max_staleness=pd.Timedelta(max_staleness)))
    assert_causal_context(ctx)
    panel = target.merge(ctx, on="timestamp", how="left", validate="one_to_one")
    context_features = [c for c in ctx.columns if c != "timestamp" and not c.endswith("__source_timestamp") and not c.endswith("__age_seconds")]
    experiments = {"target_only": target_features}
    for root in contexts:
        root_cols = [c for c in context_features if c.startswith(root + "__")]
        experiments[f"target_plus_{root}"] = target_features + root_cols
    experiments["target_plus_all_context"] = target_features + context_features
    scores = {name: _score_panel(panel, feats, label, min_train, test_size, purge_rows) for name, feats in experiments.items()}
    base = scores["target_only"]
    for score in scores.values():
        score["delta_balanced_accuracy_vs_target_only"] = score["balanced_accuracy"] - base["balanced_accuracy"]
        score["delta_roc_auc_vs_target_only"] = None if score["roc_auc"] is None or base["roc_auc"] is None else score["roc_auc"] - base["roc_auc"]
    return {"contract": "cross_market_context_ablation.v1", "selection_authority": False, "promotion_authority": False, "label": label, "target_features": target_features, "context_roots": list(contexts), "purge_rows": purge_rows, "scores": scores}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--context", action="append", default=[], help="ROOT=csv")
    ap.add_argument("--label", required=True)
    ap.add_argument("--target-feature", action="append", required=True)
    ap.add_argument("--min-train", type=int, required=True)
    ap.add_argument("--test-size", type=int, required=True)
    ap.add_argument("--purge-rows", type=int, default=0)
    ap.add_argument("--max-staleness", default="15min")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    contexts = {}
    for item in args.context:
        root, path = item.split("=", 1)
        contexts[root] = pd.read_csv(path)
    result = run_ablation(pd.read_csv(args.target), contexts, label=args.label, target_features=args.target_feature, min_train=args.min_train, test_size=args.test_size, purge_rows=args.purge_rows, max_staleness=args.max_staleness)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
