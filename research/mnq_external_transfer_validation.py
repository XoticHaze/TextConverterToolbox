from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.licensed_mnq_expanded_validation import _bars as licensed_bars, _load_aggregate, _roll_schedule, _stitch
from research.licensed_mnq_trust_gate import LOCAL_STATE, _base_model, _folds, _gate_model

PINNED_DEEP_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
HORIZON = 12
CONTRACT_RE = re.compile(r"^MNQ (?P<month>\d{2})-(?P<year>\d{2})$")


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def contract_key(name: str) -> tuple[int, int]:
    m = CONTRACT_RE.fullmatch(name)
    if not m:
        raise ValueError(name)
    return 2000 + int(m.group("year")), int(m.group("month"))


def load_deep(root: Path) -> pd.DataFrame:
    nodes = []
    files = sorted(root.glob("MNQ */*.Last.csv"))
    if not files:
        raise RuntimeError(f"no Last.csv files under {root}")
    for path in files:
        contract = path.parent.name
        if not CONTRACT_RE.fullmatch(contract):
            continue
        f = pd.read_csv(path)
        expected = ["datetime", "open", "high", "low", "close", "volume"]
        if list(f.columns) != expected:
            raise RuntimeError(f"{path}: unexpected schema {list(f.columns)}")
        if f.empty:
            continue
        f["timestamp"] = pd.to_datetime(f["datetime"], utc=True, errors="raise")
        for c in ["open", "high", "low", "close", "volume"]:
            f[c] = pd.to_numeric(f[c], errors="raise")
        f["symbol"] = contract
        nodes.append(f[["timestamp", "open", "high", "low", "close", "volume", "symbol"]])
    if not nodes:
        raise RuntimeError("no usable deep MNQ rows")
    out = pd.concat(nodes, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return out


def deep_roll_schedule(frame: pd.DataFrame, confirmation_sessions: int = 2) -> pd.DataFrame:
    work = frame.copy()
    work["session"] = work["timestamp"].dt.strftime("%Y%m%d")
    daily = work.groupby(["session", "symbol"], as_index=False)["volume"].sum()
    contracts = sorted(daily["symbol"].unique(), key=contract_key)
    sessions = sorted(daily["session"].unique())
    volume = {(r.session, r.symbol): float(r.volume) for r in daily.itertuples()}
    active = 0
    streak = 0
    pending = False
    rows = []
    for session in sessions:
        reason = "hold"
        if pending and active + 1 < len(contracts):
            active += 1; streak = 0; pending = False; reason = "volume_crossover_confirmed_prior_session"
        while active + 1 < len(contracts) and volume.get((session, contracts[active]), 0.0) <= 0 and volume.get((session, contracts[active + 1]), 0.0) > 0:
            active += 1; streak = 0; reason = "current_contract_unavailable"
        current = contracts[active]
        current_volume = volume.get((session, current), 0.0)
        next_contract = contracts[active + 1] if active + 1 < len(contracts) else None
        next_volume = volume.get((session, next_contract), 0.0) if next_contract else 0.0
        if current_volume <= 0:
            later = [c for c in contracts[active + 1:] if volume.get((session, c), 0.0) > 0]
            if later:
                active = contracts.index(later[0]); current = contracts[active]
                current_volume = volume.get((session, current), 0.0)
                next_contract = contracts[active + 1] if active + 1 < len(contracts) else None
                next_volume = volume.get((session, next_contract), 0.0) if next_contract else 0.0
                streak = 0; reason = "later_contract_availability_fallback"
        if current_volume > 0 and next_volume > current_volume:
            streak += 1
            if streak >= confirmation_sessions:
                pending = True
        elif current_volume > 0:
            streak = 0
        rows.append({"session": session, "selected_contract": current, "roll_reason": reason})
    out = pd.DataFrame(rows)
    keys = [contract_key(s) for s in out["selected_contract"]]
    if keys != sorted(keys):
        raise RuntimeError("deep MNQ roll schedule moved backward")
    return out


def stitch_deep(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["session"] = work["timestamp"].dt.strftime("%Y%m%d")
    selected = schedule.set_index("session")["selected_contract"].to_dict()
    work["selected_contract"] = work["session"].map(selected)
    work = work[work["symbol"] == work["selected_contract"]].copy()
    work = work.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
    if work["timestamp"].duplicated().any() or not work["timestamp"].is_monotonic_increasing:
        raise RuntimeError("deep MNQ stitched timestamps not unique/increasing")
    return work


def deep_bars(stitched: pd.DataFrame) -> pd.DataFrame:
    w = stitched.set_index("timestamp")
    bars = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), observed_minutes=("close", "count"),
        source_contract=("symbol", "first"), source_contract_last=("symbol", "last"),
    )
    bars = bars[bars["observed_minutes"] > 0].copy()
    mixed = bars["source_contract"] != bars["source_contract_last"]
    bars = bars.loc[~mixed].drop(columns=["source_contract_last"]).reset_index()
    bars["market"] = "MNQ"
    bars.attrs["dropped_roll_boundary_bars"] = int(mixed.sum())
    return bars


def class_thresholds(prob: np.ndarray, base: np.ndarray) -> dict[int, dict[str, float]]:
    out = {}
    for cls in (0, 1):
        values = prob[base == cls]
        if len(values) < 1000:
            raise RuntimeError(f"insufficient licensed OOF gate rows for base class {cls}: {len(values)}")
        out[cls] = {"low": float(np.quantile(values, .25)), "high": float(np.quantile(values, .65))}
    return out


def policy(y: np.ndarray, base: np.ndarray, prob: np.ndarray, thresholds: dict[int, dict[str, float]], fwd_ret: np.ndarray) -> dict:
    low = np.array([thresholds[int(cls)]["low"] for cls in base], dtype=float)
    high = np.array([thresholds[int(cls)]["high"] for cls in base], dtype=float)
    invert = prob <= low
    trust = prob >= high
    selected = invert | trust
    pred = base.copy(); pred[invert] = 1 - pred[invert]
    out = {
        "coverage": float(selected.mean()),
        "selected_rows": int(selected.sum()),
        "trust_rows": int(trust.sum()),
        "invert_rows": int(invert.sum()),
        "selected_base_class_counts": {str(cls): int((selected & (base == cls)).sum()) for cls in (0, 1)},
    }
    if selected.sum() >= 200:
        out["selected"] = metric(y[selected], pred[selected])
        signed = np.where(pred[selected] == 1, 1.0, -1.0) * fwd_ret[selected]
        out["economic_sensitivity"] = {
            "gross_mean_forward_return": float(np.mean(signed)),
            "gross_median_forward_return": float(np.median(signed)),
            "gross_positive_rate": float(np.mean(signed > 0)),
            "net_mean_after_2bp": float(np.mean(signed - 0.0002)),
            "net_mean_after_5bp": float(np.mean(signed - 0.0005)),
            "net_mean_after_10bp": float(np.mean(signed - 0.0010)),
            "note": "overlapping H12 research signals; this is not executable strategy PnL",
        }
    return out


def block_ci(y: np.ndarray, pred: np.ndarray, selected: np.ndarray, block: int = 50, reps: int = 300) -> dict[str, float] | None:
    idx = np.flatnonzero(selected)
    if len(idx) < 500:
        return None
    # Bootstrap contiguous blocks from the already-selected sequence; deterministic seed.
    ys = y[idx]; ps = pred[idx]
    n = len(idx); rng = np.random.default_rng(42)
    vals = []
    starts = np.arange(max(1, n - block + 1))
    for _ in range(reps):
        draw = []
        while len(draw) < n:
            s = int(rng.choice(starts)); draw.extend(range(s, min(s + block, n)))
        d = np.array(draw[:n], dtype=int)
        vals.append(balanced_accuracy_score(ys[d], ps[d]))
    return {"p025": float(np.quantile(vals, .025)), "median": float(np.quantile(vals, .5)), "p975": float(np.quantile(vals, .975)), "block_rows": block, "reps": reps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--licensed-aggregate", type=Path, required=True)
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    # TRAINING / POLICY SELECTION SOURCE: licensed 2022-2025 only.
    lic_raw = _load_aggregate(args.licensed_aggregate)
    lic_bars = licensed_bars(_stitch(lic_raw, _roll_schedule(lic_raw)))
    lic = _add_features(lic_bars)
    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES))
    base_features = list(dict.fromkeys(expanded + REGIME_FEATURES))
    gate_features = list(dict.fromkeys(base_features + LOCAL_STATE))
    cols = list(dict.fromkeys(["timestamp", "close", *base_features, *gate_features, "target_dir_h12"]))
    work = lic[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    folds = _folds(len(work), HORIZON)
    X = work[base_features].to_numpy(float); y = work["target_dir_h12"].astype(int).to_numpy()

    meta_parts = []
    for fold, (s, e, ts, te) in enumerate(folds[:3]):
        base_model = _base_model().fit(X[s:e], y[s:e])
        pred = base_model.predict(X[ts:te]).astype(int)
        up = base_model.predict_proba(X[ts:te])[:, 1]
        part = work.iloc[ts:te][["timestamp", *gate_features]].copy()
        part["base_pred"] = pred; part["base_confidence"] = np.abs(up - .5) * 2.0
        part["correct"] = (pred == y[ts:te]).astype(int); part["fold"] = fold
        meta_parts.append(part)
    meta = pd.concat(meta_parts, ignore_index=True)
    gate_cols = [*gate_features, "base_confidence", "base_pred"]
    separate = {}
    train_sep = np.empty(len(meta), dtype=float)
    for cls in (0, 1):
        mask = meta["base_pred"].to_numpy(int) == cls
        gate = _gate_model().fit(meta.loc[mask, gate_cols].to_numpy(float), meta.loc[mask, "correct"].to_numpy(int))
        train_sep[mask] = gate.predict_proba(meta.loc[mask, gate_cols].to_numpy(float))[:, 1]
        separate[cls] = gate
    thresholds = class_thresholds(train_sep, meta["base_pred"].to_numpy(int))
    final_base = _base_model().fit(X, y)

    # EXTERNAL SOURCE / TIME: deep public corpus; no tuning or threshold changes below this line.
    deep_raw = load_deep(args.deep_root)
    deep_schedule = deep_roll_schedule(deep_raw)
    deep_stitched = stitch_deep(deep_raw, deep_schedule)
    dbars = deep_bars(deep_stitched)
    deep = _add_features(dbars)
    deep["fwd_ret_h12"] = deep["close"].shift(-HORIZON) / deep["close"] - 1.0
    dcols = list(dict.fromkeys(["timestamp", "close", "fwd_ret_h12", *base_features, *gate_features, "target_dir_h12"]))
    dwork = deep[dcols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    periods = {
        "pre_licensed": (pd.Timestamp("2019-05-01", tz="UTC"), pd.Timestamp("2022-11-17", tz="UTC")),
        "post_licensed": (pd.Timestamp("2025-11-13", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
    }
    results = {}
    for name, (start, end) in periods.items():
        part = dwork[(dwork["timestamp"] >= start) & (dwork["timestamp"] < end)].copy()
        if len(part) < 1000:
            raise RuntimeError(f"{name}: insufficient external rows {len(part)}")
        DX = part[base_features].to_numpy(float)
        dy = part["target_dir_h12"].astype(int).to_numpy()
        base_pred = final_base.predict(DX).astype(int)
        up = final_base.predict_proba(DX)[:, 1]
        gate_frame = part[gate_features].copy(); gate_frame["base_confidence"] = np.abs(up - .5) * 2.0; gate_frame["base_pred"] = base_pred
        gate_prob = np.empty(len(part), dtype=float)
        for cls in (0, 1):
            mask = base_pred == cls
            gate_prob[mask] = separate[cls].predict_proba(gate_frame.loc[mask, gate_cols].to_numpy(float))[:, 1]
        low = np.array([thresholds[int(cls)]["low"] for cls in base_pred]); high = np.array([thresholds[int(cls)]["high"] for cls in base_pred])
        invert = gate_prob <= low; trust = gate_prob >= high; selected = invert | trust
        selected_pred = base_pred.copy(); selected_pred[invert] = 1 - selected_pred[invert]
        row = {
            "rows": int(len(part)),
            "first_timestamp": part["timestamp"].iloc[0].isoformat(),
            "last_timestamp": part["timestamp"].iloc[-1].isoformat(),
            "base": metric(dy, base_pred),
            "gate_correctness_auc": float(roc_auc_score((base_pred == dy).astype(int), gate_prob)),
            "policy": policy(dy, base_pred, gate_prob, thresholds, part["fwd_ret_h12"].to_numpy(float)),
            "selected_block_bootstrap_ba_ci": block_ci(dy, selected_pred, selected),
        }
        results[name] = row
        print(name, "BASE_BA", row["base"]["balanced_accuracy"], "SELECTED_BA", row["policy"].get("selected", {}).get("balanced_accuracy"), "COVERAGE", row["policy"]["coverage"])

    result = {
        "schema": "foundry.mnq_external_transfer_validation.v1",
        "research_only": True,
        "promotion_authority": False,
        "protocol": "licensed-only fit/policy freeze -> independent deep-source non-overlap evaluation",
        "licensed_source": "brandenmorris/mnq-1m-q4-2022-q4-2025-ts-ohlcv-sym@v1",
        "deep_source": f"mbytes21/MNQ_DATA@{PINNED_DEEP_COMMIT}",
        "base_feature_mode": "expanded_regime",
        "gate_mode": "full_state_separate_direction",
        "target": "target_dir_h12",
        "policy_quantiles": {"invert": 0.25, "trust": 0.65},
        "thresholds_learned_on_licensed_oof_only": {str(k): v for k, v in thresholds.items()},
        "excluded_forward_aligned_features": ["chikou_span"],
        "licensed_training_rows": int(len(work)),
        "licensed_oof_gate_rows": int(len(meta)),
        "deep_source_rows": int(len(deep_raw)),
        "deep_stitched_minutes": int(len(deep_stitched)),
        "deep_12min_bars": int(len(dbars)),
        "deep_dropped_roll_boundary_bars": int(dbars.attrs.get("dropped_roll_boundary_bars", 0)),
        "external_periods": results,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(); result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_EXTERNAL_TRANSFER_VALIDATION=PASS"); print("RECEIPT_SHA256=" + result["receipt_sha256"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
