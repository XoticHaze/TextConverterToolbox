from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_nq_domain_adaptation import fit_equal_market
from research.mnq_opportunity_target_matrix import classification, economic, model, target_columns
from research.mnq_to_nq_transfer import bars12, load_nq

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
MNQ_CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
WINDOW_STARTS = list(pd.date_range("2026-02-01", "2026-04-12", freq="7D", tz="UTC"))
DATA_END = pd.Timestamp("2026-04-16", tz="UTC")
BAR = pd.Timedelta(minutes=12)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
COST = 0.0002


def utc_slot(ts: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    return ((parsed - EPOCH) // BAR).to_numpy(dtype=np.int64)


def net_returns(pred: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    selected = pred != 0
    if not selected.any():
        return np.asarray([], dtype=float)
    side = np.where(pred[selected] == 1, 1.0, -1.0)
    return side * fwd[selected] - COST


def phase_audit(ts: pd.Series, pred: np.ndarray, fwd: np.ndarray, horizon: int) -> dict:
    slots = utc_slot(ts)
    phases = {}
    for phase in range(horizon):
        mask = (slots % horizon) == phase
        r = net_returns(pred[mask], fwd[mask])
        phases[str(phase)] = {
            "signals": int(len(r)),
            "mean_net_after_2bp": float(np.mean(r)) if len(r) >= 10 else None,
            "median_net_after_2bp": float(np.median(r)) if len(r) >= 10 else None,
            "positive_rate": float(np.mean(r > 0)) if len(r) >= 10 else None,
        }
    valid = [v["mean_net_after_2bp"] for v in phases.values() if v["mean_net_after_2bp"] is not None]
    return {
        "phase_streams": phases,
        "valid_phases": int(len(valid)),
        "positive_phases": int(sum(v > 0 for v in valid)),
        "positive_phase_fraction": float(np.mean(np.asarray(valid) > 0)) if valid else None,
        "median_phase_net_after_2bp": float(np.median(valid)) if valid else None,
        "min_phase_net_after_2bp": float(np.min(valid)) if valid else None,
        "max_phase_net_after_2bp": float(np.max(valid)) if valid else None,
        "contract": f"absolute UTC 12-minute slot modulo H{horizon}; within-phase forecasts are separated by at least {horizon * 12} clock minutes; every phase reported, none selected post-hoc",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--nq-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]

    mnq_raw = load_deep(args.deep_root)
    mnq = _add_features(deep_bars(stitch_deep(mnq_raw, deep_roll_schedule(mnq_raw))))
    nq = _add_features(bars12(load_nq(args.nq_csv)))
    features = list(BASE_FEATURES)
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    mnq_label, _, _ = target_columns(mnq, horizon, vol_mult)
    nq_label, nq_fwd, _ = target_columns(nq, horizon, vol_mult)
    mnq["target"] = mnq_label
    nq["target"] = nq_label
    nq["fwd"] = nq_fwd

    mnq_train = mnq[(mnq["timestamp"] < MNQ_CUTOFF) & mnq["target"].notna()].copy()
    if len(mnq_train) < 50000:
        raise RuntimeError(f"insufficient MNQ train rows {len(mnq_train)}")

    rows = []
    for start in WINDOW_STARTS:
        end = min(start + pd.Timedelta(days=7), DATA_END)
        nq_train = nq[(nq["timestamp"] < start) & nq["target"].notna()].copy()
        test = nq[(nq["timestamp"] >= start) & (nq["timestamp"] < end) & nq["target"].notna()].copy()
        if len(nq_train) < 1000 or len(test) < 300:
            continue
        if nq_train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("NQ chronology overlap")
        y_test = test["target"].astype(int).to_numpy()
        fwd = test["fwd"].to_numpy(float)
        fitted = {
            "mnq_frozen": model().fit(mnq_train[features].to_numpy(float), mnq_train["target"].astype(int).to_numpy()),
            "nq_recent_specialist": model().fit(nq_train[features].to_numpy(float), nq_train["target"].astype(int).to_numpy()),
            "equal_market_pooled": fit_equal_market(features, mnq_train, nq_train),
        }
        arms = {}
        for arm, fitted_model in fitted.items():
            pred = fitted_model.predict(test[features].to_numpy(float)).astype(int)
            phase = phase_audit(test["timestamp"], pred, fwd, horizon)
            arms[arm] = {
                "classification": classification(y_test, pred),
                "dense_economic_diagnostic": economic(pred, fwd),
                "nonoverlap_phase_audit": phase,
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
            }
            dense = arms[arm]["dense_economic_diagnostic"]
            print(args.config_key, start.date(), arm, "DENSE_NET2", None if dense is None else dense["net_mean_after_2bp"], "PHASE_MEDIAN_NET2", phase["median_phase_net_after_2bp"], "POS_PHASES", phase["positive_phases"], "/", phase["valid_phases"])
        rows.append({
            "start": start.isoformat(), "end": end.isoformat(),
            "nq_train_rows": int(len(nq_train)), "test_rows": int(len(test)),
            "nq_train_last_timestamp": nq_train["timestamp"].max().isoformat(),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "arms": arms,
        })
    if len(rows) < 8:
        raise RuntimeError(f"insufficient weekly windows {len(rows)}")

    summaries = {}
    for arm in ("mnq_frozen", "nq_recent_specialist", "equal_market_pooled"):
        med = np.asarray([r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_2bp"] for r in rows], dtype=float)
        frac = np.asarray([r["arms"][arm]["nonoverlap_phase_audit"]["positive_phase_fraction"] for r in rows], dtype=float)
        summaries[arm] = {
            "windows": len(rows),
            "windows_positive_median_nonoverlap_phase": int(np.sum(med > 0)),
            "median_of_weekly_phase_medians_net2": float(np.median(med)),
            "mean_weekly_phase_median_net2": float(np.mean(med)),
            "min_weekly_phase_median_net2": float(np.min(med)),
            "max_weekly_phase_median_net2": float(np.max(med)),
            "mean_positive_phase_fraction": float(np.mean(frac)),
        }

    result = {
        "schema": "foundry.mnq_nq_pooled_nonoverlap_weekly.v2",
        "research_only": True,
        "promotion_authority": False,
        "hypothesis_origin": "domain-adaptation discovery showed positive March-April NQ economics for baseline20 recent-NQ and equal-market pooled arms; this run tests whether the effect is temporally broad and non-overlapping rather than assuming the discovery holdout is confirmatory",
        "mnq_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "nq_source": "axb0306/cme-futures-ohlc@60abd3fb6369c6ce0b6a4a65b0f2562fc96b1264:NQ/NQ_1min_20260120_20260415.csv",
        "config_key": args.config_key, "feature_set": "baseline20", "horizon": horizon, "vol_multiplier": vol_mult,
        "cost_per_signal": COST,
        "protocol": "fixed seven-day windows from 2026-02-01; at every cutoff compare MNQ-frozen, recent-NQ specialist, and equal-total-market-weight pooled training using only prior rows; all arms share identical future timestamps; primary economics reported across every non-overlapping absolute-UTC phase stream with no phase selection",
        "mnq_train_rows": int(len(mnq_train)), "weekly_windows": rows, "summary": summaries,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_NQ_POOLED_NONOVERLAP_WEEKLY=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
