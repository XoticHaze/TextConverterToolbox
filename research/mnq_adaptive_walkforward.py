from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.licensed_mnq_trust_gate import LOCAL_STATE, _base_model, _gate_model
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars

HORIZON = 12
POLICY_LOW_Q = 0.25
POLICY_HIGH_Q = 0.65


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def fit_base(X: np.ndarray, y: np.ndarray):
    return _base_model().fit(X, y)


def inner_oof(train: pd.DataFrame, features: list[str], gate_features: list[str], target: str) -> pd.DataFrame:
    n = len(train)
    first = n // 2
    size = (n - first) // 3
    if size < 750:
        raise RuntimeError(f"inner OOF block too small: {size}")
    parts = []
    X = train[features].to_numpy(float); y = train[target].astype(int).to_numpy()
    for fold in range(3):
        test_start = first + fold * size
        test_end = n if fold == 2 else first + (fold + 1) * size
        train_end = test_start - HORIZON
        if train_end < 3000:
            raise RuntimeError("inner OOF training history too short")
        model = fit_base(X[:train_end], y[:train_end])
        pred = model.predict(X[test_start:test_end]).astype(int)
        up = model.predict_proba(X[test_start:test_end])[:, 1]
        node = train.iloc[test_start:test_end][["timestamp", *gate_features]].copy()
        node["base_pred"] = pred
        node["base_confidence"] = np.abs(up - 0.5) * 2.0
        node["correct"] = (pred == y[test_start:test_end]).astype(int)
        node["truth"] = y[test_start:test_end]
        node["fold"] = fold
        parts.append(node)
    return pd.concat(parts, ignore_index=True)


def fit_direction_gates(meta: pd.DataFrame, gate_cols: list[str]):
    gates = {}
    train_prob = np.empty(len(meta), dtype=float)
    thresholds = {}
    pred_class = meta["base_pred"].to_numpy(int)
    for cls in (0, 1):
        mask = pred_class == cls
        if mask.sum() < 500:
            raise RuntimeError(f"insufficient gate meta rows for predicted class {cls}: {mask.sum()}")
        correct = meta.loc[mask, "correct"].to_numpy(int)
        if len(np.unique(correct)) < 2:
            raise RuntimeError(f"gate correctness collapsed for predicted class {cls}")
        gate = _gate_model().fit(meta.loc[mask, gate_cols].to_numpy(float), correct)
        prob = gate.predict_proba(meta.loc[mask, gate_cols].to_numpy(float))[:, 1]
        train_prob[mask] = prob
        gates[cls] = gate
        thresholds[cls] = {"low": float(np.quantile(prob, POLICY_LOW_Q)), "high": float(np.quantile(prob, POLICY_HIGH_Q))}
    return gates, thresholds


def apply_gate(test: pd.DataFrame, base_pred: np.ndarray, base_up: np.ndarray, gates, thresholds, gate_features: list[str]):
    node = test[[*gate_features]].copy()
    node["base_confidence"] = np.abs(base_up - 0.5) * 2.0
    node["base_pred"] = base_pred
    gate_cols = [*gate_features, "base_confidence", "base_pred"]
    prob = np.empty(len(test), dtype=float)
    for cls in (0, 1):
        mask = base_pred == cls
        prob[mask] = gates[cls].predict_proba(node.loc[mask, gate_cols].to_numpy(float))[:, 1]
    low = np.array([thresholds[int(cls)]["low"] for cls in base_pred], dtype=float)
    high = np.array([thresholds[int(cls)]["high"] for cls in base_pred], dtype=float)
    invert = prob <= low
    trust = prob >= high
    selected = invert | trust
    pred = base_pred.copy(); pred[invert] = 1 - pred[invert]
    return pred, selected, trust, invert, prob


def economic(pred: np.ndarray, fwd: np.ndarray, mask: np.ndarray) -> dict | None:
    if mask.sum() < 100:
        return None
    signed = np.where(pred[mask] == 1, 1.0, -1.0) * fwd[mask]
    return {
        "rows": int(mask.sum()),
        "gross_mean_forward_return": float(np.mean(signed)),
        "gross_median_forward_return": float(np.median(signed)),
        "gross_positive_rate": float(np.mean(signed > 0)),
        "net_mean_after_2bp": float(np.mean(signed - 0.0002)),
        "net_mean_after_5bp": float(np.mean(signed - 0.0005)),
        "net_mean_after_10bp": float(np.mean(signed - 0.0010)),
        "note": "overlapping H12 research signals; sensitivity only, not executable strategy PnL",
    }


def quarter_starts() -> list[pd.Timestamp]:
    return list(pd.date_range("2022-01-01", "2026-04-01", freq="QS", tz="UTC"))


def summarize(rows: list[dict], key: str) -> dict:
    vals = [r[key]["balanced_accuracy"] for r in rows if r.get(key)]
    return {
        "quarters": len(vals),
        "mean_ba": float(np.mean(vals)),
        "median_ba": float(np.median(vals)),
        "min_ba": float(np.min(vals)),
        "max_ba": float(np.max(vals)),
        "quarters_above_0_5": int(sum(v > 0.5 for v in vals)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--deep-root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    frame["fwd_ret_h12"] = frame["close"].shift(-HORIZON) / frame["close"] - 1.0

    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES))
    expanded_regime = list(dict.fromkeys(expanded + REGIME_FEATURES))
    gate_features = list(dict.fromkeys(expanded_regime + LOCAL_STATE))
    target = "target_dir_h12"
    cols = list(dict.fromkeys(["timestamp", "fwd_ret_h12", *BASE_FEATURES, *expanded_regime, *gate_features, target]))
    work = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    quarters = quarter_starts()
    rows = []
    aggregate = {"baseline": {"truth": [], "pred": []}, "expanded": {"truth": [], "pred": []}, "gate": {"truth": [], "pred": [], "selected": [], "fwd": []}}

    for i in range(len(quarters) - 1):
        start, end = quarters[i], quarters[i + 1]
        test_idx = np.flatnonzero(((work["timestamp"] >= start) & (work["timestamp"] < end)).to_numpy())
        if len(test_idx) < 500:
            continue
        test_start = int(test_idx[0]); test_end = int(test_idx[-1] + 1)
        train_end = test_start - HORIZON
        if train_end < 10000:
            continue
        train = work.iloc[:train_end].copy(); test = work.iloc[test_start:test_end].copy()
        y_train = train[target].astype(int).to_numpy(); y_test = test[target].astype(int).to_numpy(); fwd = test["fwd_ret_h12"].to_numpy(float)

        # Fixed direct baselines.
        base20 = fit_base(train[BASE_FEATURES].to_numpy(float), y_train)
        pred20 = base20.predict(test[BASE_FEATURES].to_numpy(float)).astype(int)
        expanded_model = fit_base(train[expanded_regime].to_numpy(float), y_train)
        pred_exp = expanded_model.predict(test[expanded_regime].to_numpy(float)).astype(int)
        up_exp = expanded_model.predict_proba(test[expanded_regime].to_numpy(float))[:, 1]

        # Gate is trained only on inner OOF correctness from this quarter's historical training window.
        meta = inner_oof(train, expanded_regime, gate_features, target)
        gate_cols = [*gate_features, "base_confidence", "base_pred"]
        gates, thresholds = fit_direction_gates(meta, gate_cols)
        gated_pred, selected, trust, invert, gate_prob = apply_gate(test, pred_exp, up_exp, gates, thresholds, gate_features)

        row = {
            "period": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_last_timestamp": train["timestamp"].iloc[-1].isoformat(),
            "test_first_timestamp": test["timestamp"].iloc[0].isoformat(),
            "test_last_timestamp": test["timestamp"].iloc[-1].isoformat(),
            "baseline20": metric(y_test, pred20),
            "expanded_regime": metric(y_test, pred_exp),
            "gate_selected": metric(y_test[selected], gated_pred[selected]) if selected.sum() >= 100 else None,
            "gate_coverage": float(selected.mean()),
            "trust_rows": int(trust.sum()),
            "invert_rows": int(invert.sum()),
            "gate_selected_class_counts": {str(cls): int((selected & (gated_pred == cls)).sum()) for cls in (0,1)},
            "economic": {
                "baseline20": economic(pred20, fwd, np.ones(len(test), dtype=bool)),
                "expanded_regime": economic(pred_exp, fwd, np.ones(len(test), dtype=bool)),
                "gate_selected": economic(gated_pred, fwd, selected),
            },
        }
        rows.append(row)
        aggregate["baseline"]["truth"].extend(y_test.tolist()); aggregate["baseline"]["pred"].extend(pred20.tolist())
        aggregate["expanded"]["truth"].extend(y_test.tolist()); aggregate["expanded"]["pred"].extend(pred_exp.tolist())
        aggregate["gate"]["truth"].extend(y_test.tolist()); aggregate["gate"]["pred"].extend(gated_pred.tolist()); aggregate["gate"]["selected"].extend(selected.tolist()); aggregate["gate"]["fwd"].extend(fwd.tolist())
        print(row["period"], "BASE20", row["baseline20"]["balanced_accuracy"], "EXP", row["expanded_regime"]["balanced_accuracy"], "GATE", None if row["gate_selected"] is None else row["gate_selected"]["balanced_accuracy"], "COV", row["gate_coverage"])

    if len(rows) < 8:
        raise RuntimeError(f"insufficient outer quarters: {len(rows)}")
    yb=np.array(aggregate["baseline"]["truth"],int); pb=np.array(aggregate["baseline"]["pred"],int)
    ye=np.array(aggregate["expanded"]["truth"],int); pe=np.array(aggregate["expanded"]["pred"],int)
    yg=np.array(aggregate["gate"]["truth"],int); pg=np.array(aggregate["gate"]["pred"],int); sg=np.array(aggregate["gate"]["selected"],bool); fg=np.array(aggregate["gate"]["fwd"],float)
    result = {
        "schema": "foundry.mnq_adaptive_walkforward.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "protocol": "quarterly expanding past-only refit; H12 purge; inner chronological OOF gate correctness; fixed q25 invert/q65 trust policy",
        "target": target,
        "excluded_forward_aligned_features": ["chikou_span"],
        "deep_source_rows": int(len(raw)),
        "stitched_minutes": int(len(stitched)),
        "bars_12min": int(len(bars)),
        "walkforward_rows": rows,
        "stability": {
            "baseline20": summarize(rows, "baseline20"),
            "expanded_regime": summarize(rows, "expanded_regime"),
            "gate_selected": summarize(rows, "gate_selected"),
            "gate_improved_vs_expanded_quarters": int(sum(r["gate_selected"] is not None and r["gate_selected"]["balanced_accuracy"] > r["expanded_regime"]["balanced_accuracy"] for r in rows)),
        },
        "aggregate": {
            "baseline20": metric(yb,pb),
            "expanded_regime": metric(ye,pe),
            "gate_selected": metric(yg[sg],pg[sg]),
            "gate_coverage": float(sg.mean()),
            "gate_economic": economic(pg,fg,sg),
        },
    }
    material=json.dumps(result,sort_keys=True,separators=(",",":")).encode(); result["receipt_sha256"]=hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("MNQ_ADAPTIVE_WALKFORWARD=PASS"); print("RECEIPT_SHA256="+result["receipt_sha256"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
