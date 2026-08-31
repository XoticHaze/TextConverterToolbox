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
from research.mnq_opportunity_target_matrix import classification, model, target_columns
from research.nq_cc0_multiyear_validation import load_normalized, bars12

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
NQ_CUTOFF = pd.Timestamp("2025-12-12", tz="UTC")
MNQ_CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
MNQ_TEST_END = pd.Timestamp("2026-03-01", tz="UTC")
BAR = pd.Timedelta(minutes=12)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
MNQ_DOLLARS_PER_POINT = 2.0
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
ARMS = ("nq_frozen", "mnq_local_frozen", "equal_market_pooled")


def utc_slot(ts: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(ts, utc=True, errors="raise")
    return ((parsed - EPOCH) // BAR).to_numpy(dtype=np.int64)


def signed_points(pred: np.ndarray, point_move: np.ndarray) -> np.ndarray:
    selected = pred != 0
    if not selected.any():
        return np.asarray([], dtype=float)
    side = np.where(pred[selected] == 1, 1.0, -1.0)
    return side * point_move[selected]


def point_economics(pred: np.ndarray, point_move: np.ndarray) -> dict | None:
    gross = signed_points(pred, point_move)
    if len(gross) < 100:
        return None
    result = {
        "signals": int(len(gross)),
        "gross_mean_points": float(np.mean(gross)),
        "gross_median_points": float(np.median(gross)),
        "gross_positive_rate": float(np.mean(gross > 0)),
    }
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        net = gross - cost
        result[f"net_mean_points_after_{key}pt"] = float(np.mean(net))
        result[f"net_mean_mnq_dollars_after_{key}pt"] = float(np.mean(net) * MNQ_DOLLARS_PER_POINT)
    return result


def phase_audit(ts: pd.Series, pred: np.ndarray, point_move: np.ndarray, horizon: int) -> dict:
    slots = utc_slot(ts)
    phases: dict[str, dict] = {}
    for phase in range(horizon):
        mask = (slots % horizon) == phase
        gross = signed_points(pred[mask], point_move[mask])
        rec = {"signals": int(len(gross))}
        if len(gross) >= 10:
            rec["gross_mean_points"] = float(np.mean(gross))
            for cost in POINT_COSTS:
                key = str(cost).replace(".", "p")
                mean_points = float(np.mean(gross - cost))
                rec[f"net_mean_points_after_{key}pt"] = mean_points
                rec[f"net_mean_mnq_dollars_after_{key}pt"] = mean_points * MNQ_DOLLARS_PER_POINT
        phases[str(phase)] = rec
    valid = [v for v in phases.values() if "net_mean_points_after_1p0pt" in v]
    def arr(key: str) -> np.ndarray:
        return np.asarray([float(v[key]) for v in valid], dtype=float)
    out = {
        "phase_streams": phases,
        "valid_phases": int(len(valid)),
        "contract": f"absolute UTC 12-minute slot modulo H{horizon}; within-phase forecasts separated by at least {horizon * 12} minutes; every phase reported, none selected post-hoc",
    }
    if valid:
        for cost in POINT_COSTS:
            key = str(cost).replace(".", "p")
            vals = arr(f"net_mean_points_after_{key}pt")
            out[f"median_phase_net_points_after_{key}pt"] = float(np.median(vals))
            out[f"mean_phase_net_points_after_{key}pt"] = float(np.mean(vals))
            out[f"positive_phase_fraction_after_{key}pt"] = float(np.mean(vals > 0))
            out[f"median_phase_net_mnq_dollars_after_{key}pt"] = float(np.median(vals) * MNQ_DOLLARS_PER_POINT)
    return out


def weekly_audit(test: pd.DataFrame, pred: np.ndarray, point_move: np.ndarray, horizon: int) -> list[dict]:
    temp = test[["timestamp"]].copy()
    temp["pred"] = pred
    temp["point_move"] = point_move
    t = pd.to_datetime(temp["timestamp"], utc=True)
    temp["week_start"] = t.dt.normalize() - pd.to_timedelta(t.dt.weekday, unit="D")
    rows = []
    for week_start, g in temp.groupby("week_start", sort=True):
        if len(g) < 100:
            continue
        phase = phase_audit(g["timestamp"], g["pred"].to_numpy(int), g["point_move"].to_numpy(float), horizon)
        rows.append({
            "week_start": week_start.isoformat(),
            "rows": int(len(g)),
            "phase_audit": phase,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--nq-normalized", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]
    features = list(BASE_FEATURES)

    mnq_raw = load_deep(args.deep_root)
    mnq = _add_features(deep_bars(stitch_deep(mnq_raw, deep_roll_schedule(mnq_raw))))
    nq_raw = load_normalized(args.nq_normalized)
    nq = _add_features(bars12(nq_raw))

    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    mnq_label, _, _ = target_columns(mnq, horizon, vol_mult)
    nq_label, _, _ = target_columns(nq, horizon, vol_mult)
    mnq["target"] = mnq_label
    nq["target"] = nq_label
    mnq["point_move"] = mnq["close"].shift(-horizon) - mnq["close"]

    # Purge the final H bars before each frozen boundary so no training label reaches into test time.
    mnq_before = mnq[mnq["timestamp"] < MNQ_CUTOFF].copy()
    nq_before = nq[nq["timestamp"] < NQ_CUTOFF].copy()
    if len(mnq_before) <= horizon or len(nq_before) <= horizon:
        raise RuntimeError("insufficient pre-cutoff rows")
    mnq_train = mnq_before.iloc[:-horizon].copy()
    nq_train = nq_before.iloc[:-horizon].copy()
    mnq_train = mnq_train[mnq_train["target"].notna()].copy()
    nq_train = nq_train[nq_train["target"].notna()].copy()
    mnq_test = mnq[(mnq["timestamp"] >= MNQ_CUTOFF) & (mnq["timestamp"] < MNQ_TEST_END) & mnq["target"].notna() & mnq["point_move"].notna()].copy()

    if len(mnq_train) < 50000 or len(nq_train) < 50000 or len(mnq_test) < 2500:
        raise RuntimeError(f"insufficient rows mnq_train={len(mnq_train)} nq_train={len(nq_train)} mnq_test={len(mnq_test)}")
    if mnq_train["timestamp"].max() >= mnq_test["timestamp"].min():
        raise RuntimeError("MNQ local chronology overlap")
    if nq_train["timestamp"].max() >= mnq_test["timestamp"].min():
        raise RuntimeError("NQ donor chronology overlap")

    fitted = {
        "nq_frozen": model().fit(nq_train[features].to_numpy(float), nq_train["target"].astype(int).to_numpy()),
        "mnq_local_frozen": model().fit(mnq_train[features].to_numpy(float), mnq_train["target"].astype(int).to_numpy()),
        "equal_market_pooled": fit_equal_market(features, mnq_train, nq_train),
    }
    y_test = mnq_test["target"].astype(int).to_numpy()
    point_move = mnq_test["point_move"].to_numpy(float)
    arms: dict[str, dict] = {}
    for arm, fitted_model in fitted.items():
        pred = fitted_model.predict(mnq_test[features].to_numpy(float)).astype(int)
        arms[arm] = {
            "classification": classification(y_test, pred),
            "dense_point_economic_diagnostic": point_economics(pred, point_move),
            "nonoverlap_phase_audit": phase_audit(mnq_test["timestamp"], pred, point_move, horizon),
            "weekly_nonoverlap_audit": weekly_audit(mnq_test, pred, point_move, horizon),
            "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
        }
        phase = arms[arm]["nonoverlap_phase_audit"]
        print(args.config_key, arm, "BA", arms[arm]["classification"]["balanced_accuracy"], "MEDIAN_PHASE_NET_1PT", phase.get("median_phase_net_points_after_1p0pt"), "MNQ_DOLLARS", phase.get("median_phase_net_mnq_dollars_after_1p0pt"))

    result = {
        "schema": "foundry.nq_to_mnq_execution_transfer.v1",
        "research_only": True,
        "promotion_authority": False,
        "execution_target": "MNQ",
        "information_donor": "NQ",
        "mnq_multiplier_dollars_per_point": MNQ_DOLLARS_PER_POINT,
        "cost_point_grid": list(POINT_COSTS),
        "mnq_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "nq_source": "tgtanalytics/nq-futures-1min-bar-2022-2025 CC0; exact normalized input SHA recorded below",
        "nq_normalized_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "feature_set": "baseline20",
        "protocol": "fit frozen NQ-only donor, frozen MNQ-local, and equal-total-market-weight pooled models strictly before 2026; compare them on identical deep-MNQ 2026 timestamps; horizon purge before cutoffs; no MNQ test calibration; primary economics are MNQ point/$ outcomes across all non-overlapping UTC phase streams",
        "nq_train_rows": int(len(nq_train)),
        "nq_train_last_timestamp": nq_train["timestamp"].max().isoformat(),
        "mnq_train_rows": int(len(mnq_train)),
        "mnq_train_last_timestamp": mnq_train["timestamp"].max().isoformat(),
        "mnq_test_rows": int(len(mnq_test)),
        "mnq_test_first_timestamp": mnq_test["timestamp"].min().isoformat(),
        "mnq_test_last_timestamp": mnq_test["timestamp"].max().isoformat(),
        "excluded_forward_aligned_features": ["chikou_span"],
        "arms": arms,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_TO_MNQ_EXECUTION_TRANSFER=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
