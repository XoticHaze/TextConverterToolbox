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

HORIZON = 12
VOL_MULT = 1.0
MNQ_CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
WINDOWS = [
    (pd.Timestamp("2026-02-15", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
    (pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-03-16", tz="UTC")),
    (pd.Timestamp("2026-03-16", tz="UTC"), pd.Timestamp("2026-04-16", tz="UTC")),
]
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


def phase_audit(ts: pd.Series, pred: np.ndarray, fwd: np.ndarray) -> dict:
    slots = utc_slot(ts)
    phases = {}
    for phase in range(HORIZON):
        mask = (slots % HORIZON) == phase
        r = net_returns(pred[mask], fwd[mask])
        if len(r) < 20:
            phases[str(phase)] = {"signals": int(len(r)), "mean_net_after_2bp": None}
            continue
        phases[str(phase)] = {
            "signals": int(len(r)),
            "mean_net_after_2bp": float(np.mean(r)),
            "median_net_after_2bp": float(np.median(r)),
            "positive_rate": float(np.mean(r > 0)),
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
        "contract": "each phase is an absolute UTC 12-minute slot modulo H12; signals within a phase are separated by at least 144 clock minutes; all 12 phases reported, none selected post-hoc",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--nq-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    mnq_raw = load_deep(args.deep_root)
    mnq = _add_features(deep_bars(stitch_deep(mnq_raw, deep_roll_schedule(mnq_raw))))
    nq = _add_features(bars12(load_nq(args.nq_csv)))
    features = list(BASE_FEATURES)
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    mnq_label, _, _ = target_columns(mnq, HORIZON, VOL_MULT)
    nq_label, nq_fwd, _ = target_columns(nq, HORIZON, VOL_MULT)
    mnq["target"] = mnq_label
    nq["target"] = nq_label
    nq["fwd"] = nq_fwd

    mnq_train = mnq[(mnq["timestamp"] < MNQ_CUTOFF) & mnq["target"].notna()].copy()
    if len(mnq_train) < 50000:
        raise RuntimeError(f"insufficient MNQ train rows {len(mnq_train)}")

    rows = []
    for start, end in WINDOWS:
        nq_train = nq[(nq["timestamp"] < start) & nq["target"].notna()].copy()
        test = nq[(nq["timestamp"] >= start) & (nq["timestamp"] < end) & nq["target"].notna()].copy()
        if len(nq_train) < 1500 or len(test) < 900:
            raise RuntimeError(f"insufficient NQ rows at {start}: train={len(nq_train)} test={len(test)}")
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
            arms[arm] = {
                "classification": classification(y_test, pred),
                "dense_economic_diagnostic": economic(pred, fwd),
                "nonoverlap_phase_audit": phase_audit(test["timestamp"], pred, fwd),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
            }
            dense = arms[arm]["dense_economic_diagnostic"]
            phase = arms[arm]["nonoverlap_phase_audit"]
            print(start.date(), arm, "DENSE_NET2", None if dense is None else dense["net_mean_after_2bp"], "PHASE_MEDIAN_NET2", phase["median_phase_net_after_2bp"], "POS_PHASES", phase["positive_phases"], "/", phase["valid_phases"])
        rows.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "nq_train_rows": int(len(nq_train)),
            "test_rows": int(len(test)),
            "nq_train_last_timestamp": nq_train["timestamp"].max().isoformat(),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "arms": arms,
        })

    summaries = {}
    for arm in ("mnq_frozen", "nq_recent_specialist", "equal_market_pooled"):
        phase_medians = [r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_2bp"] for r in rows]
        phase_fracs = [r["arms"][arm]["nonoverlap_phase_audit"]["positive_phase_fraction"] for r in rows]
        summaries[arm] = {
            "windows": len(rows),
            "windows_positive_median_nonoverlap_phase": int(sum(v is not None and v > 0 for v in phase_medians)),
            "median_of_window_phase_medians_net2": float(np.median(phase_medians)),
            "mean_positive_phase_fraction": float(np.mean(phase_fracs)),
        }

    result = {
        "schema": "foundry.mnq_nq_pooled_nonoverlap_validation.v1",
        "research_only": True,
        "promotion_authority": False,
        "hypothesis_origin": "follow-up to 2026-03-16 holdout where baseline20 equal-market pooled H12 had +7.84bp dense net after 2bp; this run tests temporal and non-overlap robustness rather than treating that discovery holdout as confirmatory",
        "mnq_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "nq_source": "axb0306/cme-futures-ohlc@60abd3fb6369c6ce0b6a4a65b0f2562fc96b1264:NQ/NQ_1min_20260120_20260415.csv",
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "vol_multiplier": VOL_MULT,
        "cost_per_signal": COST,
        "protocol": "three predeclared forward NQ windows; at each cutoff train NQ specialist and equal-market MNQ/NQ pool using only rows before window; fixed MNQ pre-2026 source; every arm evaluated on identical timestamps; dense economics retained only as diagnostic; primary robustness evidence is all 12 non-overlapping absolute-UTC phase streams with no post-hoc phase selection",
        "mnq_train_rows": int(len(mnq_train)),
        "windows": rows,
        "summary": summaries,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_NQ_POOLED_NONOVERLAP_VALIDATION=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
