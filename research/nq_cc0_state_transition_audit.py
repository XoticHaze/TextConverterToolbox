from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.nq_cc0_multiyear_validation import bars12, load_normalized


def ret(close: pd.Series, bars: int) -> float | None:
    if len(close) <= bars or close.iloc[-bars - 1] <= 0:
        return None
    return float(close.iloc[-1] / close.iloc[-bars - 1] - 1.0)


def state_features(history: pd.DataFrame) -> dict[str, float | None]:
    h = history.copy()
    lr = np.log(h["close"].astype(float)).diff()
    rng = (h["high"].astype(float) - h["low"].astype(float)) / h["close"].astype(float)
    vol = h["volume"].astype(float)
    close = h["close"].astype(float)
    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
    ma100 = close.rolling(100).mean().iloc[-1] if len(close) >= 100 else np.nan
    recent = vol.iloc[-120:] if len(vol) >= 120 else vol
    prior = vol.iloc[-600:-120] if len(vol) >= 600 else vol.iloc[:-len(recent)]
    return {
        "ret_1d_approx": ret(close, 120),
        "ret_5d_approx": ret(close, 600),
        "rv_1d": float(lr.iloc[-120:].std()) if len(lr) >= 120 else None,
        "rv_5d": float(lr.iloc[-600:].std()) if len(lr) >= 600 else None,
        "median_range_1d": float(rng.iloc[-120:].median()) if len(rng) >= 120 else None,
        "median_range_5d": float(rng.iloc[-600:].median()) if len(rng) >= 600 else None,
        "close_vs_ma20": float(close.iloc[-1] / ma20 - 1.0) if pd.notna(ma20) and ma20 else None,
        "close_vs_ma100": float(close.iloc[-1] / ma100 - 1.0) if pd.notna(ma100) and ma100 else None,
        "volume_recent_vs_prior": float(recent.median() / prior.median() - 1.0) if len(prior) and prior.median() > 0 else None,
        "absret_lag1_corr_5d": float(lr.abs().iloc[-600:].autocorr(lag=1)) if len(lr) >= 600 else None,
    }


def outcome_map(receipt: dict, arm: str) -> dict[str, float | None]:
    out = {}
    for row in receipt["weekly_windows"]:
        out[row["start"]] = row["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_2bp"]
    return out


def standardized_effect(values: np.ndarray, labels: np.ndarray) -> float | None:
    pos = values[labels]
    neg = values[~labels]
    if len(pos) < 5 or len(neg) < 5:
        return None
    pooled = np.sqrt((np.var(pos, ddof=1) + np.var(neg, ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return None
    return float((np.mean(pos) - np.mean(neg)) / pooled)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq-normalized", type=Path, required=True)
    ap.add_argument("--h12-receipt", type=Path, required=True)
    ap.add_argument("--h24-receipt", type=Path, required=True)
    ap.add_argument("--arm", default="rolling_90d_nq")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    h12 = json.loads(args.h12_receipt.read_text())
    h24 = json.loads(args.h24_receipt.read_text())
    m12 = outcome_map(h12, args.arm)
    m24 = outcome_map(h24, args.arm)
    common = sorted(set(m12) & set(m24))
    raw = load_normalized(args.nq_normalized)
    bars = bars12(raw)

    rows = []
    for start_s in common:
        start = pd.Timestamp(start_s)
        hist = bars[(bars["timestamp"] < start) & (bars["timestamp"] >= start - pd.Timedelta(days=35))]
        if len(hist) < 700:
            continue
        f = state_features(hist)
        rows.append({
            "start": start_s,
            "h12_net2": m12[start_s],
            "h24_net2": m24[start_s],
            "both_positive": bool(m12[start_s] is not None and m24[start_s] is not None and m12[start_s] > 0 and m24[start_s] > 0),
            "state": f,
        })

    if len(rows) < 60:
        raise RuntimeError(f"insufficient aligned state/outcome weeks {len(rows)}")
    labels = np.asarray([r["both_positive"] for r in rows], dtype=bool)
    features = sorted(rows[0]["state"])
    effects = {}
    for key in features:
        vals = np.asarray([np.nan if r["state"][key] is None else r["state"][key] for r in rows], dtype=float)
        mask = np.isfinite(vals)
        effects[key] = {
            "weeks": int(mask.sum()),
            "standardized_mean_difference_both_positive_vs_other": standardized_effect(vals[mask], labels[mask]),
            "median_when_both_positive": float(np.median(vals[mask & labels])) if np.any(mask & labels) else None,
            "median_other": float(np.median(vals[mask & ~labels])) if np.any(mask & ~labels) else None,
        }
    ranked = sorted(
        ((k, v["standardized_mean_difference_both_positive_vs_other"]) for k, v in effects.items() if v["standardized_mean_difference_both_positive_vs_other"] is not None),
        key=lambda kv: abs(kv[1]), reverse=True,
    )
    result = {
        "schema": "foundry.nq_cc0_state_transition_audit.v1",
        "research_only": True,
        "promotion_authority": False,
        "arm": args.arm,
        "protocol": "descriptive discovery only: state features use bars strictly before each weekly test start; outcomes are frozen non-overlapping H12/H24 weekly economics; no state threshold, classifier, or trading gate is fit or selected",
        "weeks": len(rows),
        "weeks_both_horizons_positive": int(labels.sum()),
        "weeks_not_both_positive": int((~labels).sum()),
        "feature_effects": effects,
        "ranked_absolute_effects": [{"feature": k, "effect": float(v)} for k, v in ranked],
        "weekly_rows": rows,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_CC0_STATE_TRANSITION_AUDIT=PASS")
    print("WEEKS=" + str(len(rows)))
    print("BOTH_POSITIVE=" + str(int(labels.sum())))
    for k, v in ranked[:8]:
        print("STATE_EFFECT", k, v)
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
