from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research import expanded_regime_ablation as ab

LOCAL_STATE = [
    "vol_ratio_12_120", "atr14_pct", "atr28_pct", "z_volume_120", "z_close_20", "z_close_60",
    "ema20_50_spread", "ema50_200_spread", "efficiency_30", "ret_skew_60", "ret_autocorr_60",
    "bb20_width", "bb20_pos", "donchian55_pos", "mfi_14", "rsi_14", "vwap_dist_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def metric(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
    }


def base_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42, C=0.5)),
    ])


def gate_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=43, C=0.25)),
    ])


def add_relational(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    wide = None
    for root, frame in frames.items():
        node = frame[["timestamp", "ret_1", "ret_6", "rv_12"]].rename(columns={
            "ret_1": f"{root}_ret1", "ret_6": f"{root}_ret6", "rv_12": f"{root}_rv12"
        })
        wide = node if wide is None else wide.merge(node, on="timestamp", how="outer")
    wide = wide.sort_values("timestamp").reset_index(drop=True)
    wide["equity_factor_6"] = wide[["MNQ_ret6", "NQ_ret6"]].mean(axis=1)
    wide["commodity_factor_6"] = wide[["CL_ret6", "GC_ret6"]].mean(axis=1)
    wide["equity_dispersion_6"] = (wide["MNQ_ret6"] - wide["NQ_ret6"]).abs()
    wide["equity_rates_spread_6"] = wide["equity_factor_6"] - wide["ZN_ret6"]
    wide["equity_commodity_spread_6"] = wide["equity_factor_6"] - wide["commodity_factor_6"]
    factor_cols = ["equity_factor_6", "commodity_factor_6", "equity_dispersion_6", "equity_rates_spread_6", "equity_commodity_spread_6"]

    out = {}
    union = set(factor_cols)
    for root, frame in frames.items():
        g = frame.merge(wide, on="timestamp", how="left", suffixes=("", "_wide"))
        rel = list(factor_cols)
        target_r1 = g["ret_1"]
        target_r6 = g["ret_6"]
        target_rv = g["rv_12"]
        for other in ab.ROOTS:
            if other == root:
                continue
            r1 = g[f"{other}_ret1"]
            r6 = g[f"{other}_ret6"]
            rv = g[f"{other}_rv12"]
            for n in (30, 120):
                corr = f"rel_{other}_corr{n}"
                g[corr] = target_r1.rolling(n, min_periods=n).corr(r1)
                rel.append(corr)
            cov = target_r1.rolling(60, min_periods=60).cov(r1)
            var = r1.rolling(60, min_periods=60).var().replace(0, np.nan)
            beta = f"rel_{other}_beta60"
            residual = f"rel_{other}_residual1"
            volratio = f"rel_{other}_volratio"
            spread6 = f"rel_{other}_spread6"
            g[beta] = cov / var
            g[residual] = target_r1 - g[beta] * r1
            g[volratio] = target_rv / rv.replace(0, np.nan)
            g[spread6] = target_r6 - r6
            rel += [beta, residual, volratio, spread6]
        union.update(rel)
        g.attrs["relational_features"] = rel
        out[root] = g
    return out, sorted(union)


def policy(y, base, prob, train_prob):
    low = float(np.quantile(train_prob, 0.25))
    high = float(np.quantile(train_prob, 0.65))
    trust = prob >= high
    invert = prob <= low
    selected = trust | invert
    row = {"coverage": float(selected.mean()), "trust_rows": int(trust.sum()), "invert_rows": int(invert.sum()), "low_cut": low, "high_cut": high}
    if selected.sum() >= 50:
        pred = base.copy(); pred[invert] = 1 - pred[invert]
        row["selected"] = metric(y[selected], pred[selected])
    if trust.sum() >= 50:
        row["trust"] = metric(y[trust], base[trust])
    if invert.sum() >= 50:
        row["invert"] = metric(y[invert], 1 - base[invert])
    return row


def prepare(root: str, frame: pd.DataFrame):
    rel = frame.attrs["relational_features"]
    all_state = list(dict.fromkeys([*ab.REGIME_FEATURES, *LOCAL_STATE, *rel]))
    cols = list(dict.fromkeys(["timestamp", *ab.BASE_FEATURES, *all_state, "target_dir_h12"]))
    w = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    folds = ab._row_folds(len(w), 12)
    X = w[ab.BASE_FEATURES].to_numpy(float); y = w["target_dir_h12"].astype(int).to_numpy()
    parts = []
    for fold, (s, e, ts, te) in enumerate(folds[:3]):
        m = base_model().fit(X[s:e], y[s:e]); pred = m.predict(X[ts:te]).astype(int); up = m.predict_proba(X[ts:te])[:, 1]
        p = w.iloc[ts:te][["timestamp", *all_state]].copy(); p["base_pred"] = pred; p["base_confidence"] = np.abs(up - .5) * 2
        p["correct"] = (pred == y[ts:te]).astype(int); p["truth"] = y[ts:te]; p["market"] = root; p["fold"] = fold; parts.append(p)
    meta = pd.concat(parts, ignore_index=True)
    _, e, ts, te = folds[3]
    m = base_model().fit(X[:e], y[:e]); pred = m.predict(X[ts:te]).astype(int); up = m.predict_proba(X[ts:te])[:, 1]
    hold = w.iloc[ts:te][["timestamp", *all_state]].copy(); hold["base_pred"] = pred; hold["base_confidence"] = np.abs(up - .5) * 2
    hold["correct"] = (pred == y[ts:te]).astype(int); hold["truth"] = y[ts:te]; hold["market"] = root
    return meta, hold, rel


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
        meta, hold, rel = prepare(root, frame); prepared[root] = (meta, hold, rel); pooled_meta.append(meta); pooled_hold.append(hold)

    result = {"schema": "foundry.h12_relational_trust_gate.v1", "research_only": True, "promotion_authority": False,
              "source_commit": ab.SOURCE_COMMIT, "target": "target_dir_h12", "relational_feature_union_count": len(relational_union), "markets": {}, "pooled": {}}
    for root, (meta, hold, rel) in prepared.items():
        modes = {
            "regime_only": list(ab.REGIME_FEATURES),
            "relational_only": rel,
            "state_relational": list(dict.fromkeys([*ab.REGIME_FEATURES, *LOCAL_STATE, *rel])),
        }
        result["markets"][root] = {}
        for name, features in modes.items():
            cols = [*features, "base_confidence", "base_pred"]
            g = gate_model().fit(meta[cols].to_numpy(float), meta["correct"].to_numpy(int))
            tp = g.predict_proba(meta[cols].to_numpy(float))[:, 1]; hp = g.predict_proba(hold[cols].to_numpy(float))[:, 1]
            row = {"feature_count": len(cols), "auc": float(roc_auc_score(hold["correct"], hp)), "base": metric(hold["truth"], hold["base_pred"]),
                   "policy": policy(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int), hp, tp)}
            result["markets"][root][name] = row
        print(root, {k: round(v["policy"].get("selected", {}).get("balanced_accuracy", 0), 4) for k, v in result["markets"][root].items()})

    meta = pd.concat(pooled_meta, ignore_index=True); hold = pd.concat(pooled_hold, ignore_index=True)
    ids = []
    for root in ab.ROOTS:
        c = f"market_{root}"; ids.append(c); meta[c] = (meta["market"] == root).astype(float); hold[c] = (hold["market"] == root).astype(float)
    pooled_modes = {
        "regime_only": list(ab.REGIME_FEATURES),
        "relational_only": relational_union,
        "state_relational": list(dict.fromkeys([*ab.REGIME_FEATURES, *LOCAL_STATE, *relational_union])),
    }
    for name, features in pooled_modes.items():
        cols = [*features, *ids, "base_confidence", "base_pred"]
        g = gate_model().fit(meta[cols].to_numpy(float), meta["correct"].to_numpy(int))
        tp = g.predict_proba(meta[cols].to_numpy(float))[:, 1]; hp = g.predict_proba(hold[cols].to_numpy(float))[:, 1]
        result["pooled"][name] = {"feature_count": len(cols), "auc": float(roc_auc_score(hold["correct"], hp)),
            "base": metric(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int)),
            "policy": policy(hold["truth"].to_numpy(int), hold["base_pred"].to_numpy(int), hp, tp)}

    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(); result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("H12_RELATIONAL_TRUST_GATE=PASS"); print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
