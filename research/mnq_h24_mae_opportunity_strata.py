from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.expanded_regime_ablation import BASE_FEATURES
from research.mnq_expected_move_axb_2026 import AXB_PIN, CUTOFF, HORIZON, load_axb_mnq, ridge_model
from research.mnq_external_transfer_validation import deep_bars, deep_roll_schedule, stitch_deep
from research.mnq_h24_mae_axb_transfer import _corr, _prepare, _spearman
from research.mnq_h24_mae_risk_specialist import risk_model

MIN_TEST_ROWS = 3000
MIN_STRATUM_ROWS = 500
MIN_TAIL_ROWS = 50
OPPORTUNITY_QUANTILES = (0.25, 0.50, 0.75)
RISK_TAIL_QUANTILES = (0.25, 0.75)
MIN_POSITIVE_STRATA = 3


def _add_move_target(frame):
    x = frame.copy()
    scale = x["close"].astype(float) * x["rv_120"].astype(float) * math.sqrt(HORIZON)
    x["target_move_z"] = (x["close"].shift(-HORIZON) - x["close"]) / scale.replace(0, np.nan)
    return x


def _oof_predictions(x, risk_y, move_y):
    n = len(risk_y)
    first = n // 2
    fold = (n - first) // 4
    risk_pred = []
    move_pred = []
    risk_truth = []
    folds = []
    for i in range(4):
        ts = first + i * fold
        te = n if i == 3 else first + (i + 1) * fold
        train_end = ts - HORIZON
        if train_end < 5000 or te - ts < 1000:
            raise RuntimeError(f"invalid OOF fold {i}")
        rm = risk_model(); rm.fit(x[:train_end], risk_y[:train_end])
        em = ridge_model(); em.fit(x[:train_end], move_y[:train_end])
        risk_pred.append(np.asarray(rm.predict(x[ts:te]), dtype=float))
        move_pred.append(np.asarray(em.predict(x[ts:te]), dtype=float))
        risk_truth.append(np.asarray(risk_y[ts:te], dtype=float))
        folds.append({"fold": i, "train_rows": int(train_end), "test_rows": int(te-ts)})
    return np.concatenate(risk_pred), np.concatenate(move_pred), np.concatenate(risk_truth), folds


def _index(values, cuts):
    return np.digitize(np.asarray(values, dtype=float), np.asarray(cuts, dtype=float), right=False)


def _training_contract(risk_pred, side_opp, realized):
    opp_cuts = np.quantile(side_opp, OPPORTUNITY_QUANTILES).astype(float)
    if not np.all(np.diff(opp_cuts) > 0):
        raise RuntimeError(f"non-distinct opportunity cuts {opp_cuts.tolist()}")
    opp_idx = _index(side_opp, opp_cuts)
    strata = []
    for s in range(4):
        mask = opp_idx == s
        if int(mask.sum()) < MIN_STRATUM_ROWS:
            raise RuntimeError(f"OOF opportunity stratum {s+1} has only {int(mask.sum())} rows")
        risk_cuts = np.quantile(risk_pred[mask], RISK_TAIL_QUANTILES).astype(float)
        if not np.all(np.diff(risk_cuts) > 0):
            raise RuntimeError(f"non-distinct risk cuts in opportunity stratum {s+1}")
        strata.append({
            "stratum": s + 1,
            "oof_rows": int(mask.sum()),
            "side_aligned_ridge_min": None if s == 0 else float(opp_cuts[s-1]),
            "side_aligned_ridge_max_exclusive": None if s == 3 else float(opp_cuts[s]),
            "predicted_mae_z_q25": float(risk_cuts[0]),
            "predicted_mae_z_q75": float(risk_cuts[1]),
            "oof_risk_vs_realized_spearman": _spearman(risk_pred[mask], realized[mask]),
            "oof_opportunity_vs_realized_mae_spearman": _spearman(side_opp[mask], realized[mask]),
        })
    return {
        "opportunity_quantiles": list(OPPORTUNITY_QUANTILES),
        "side_aligned_ridge_cuts": [float(v) for v in opp_cuts],
        "conditional_risk_tail_quantiles": list(RISK_TAIL_QUANTILES),
        "strata": strata,
    }


def _test_strata(risk_pred, side_opp, realized, contract):
    opp_idx = _index(side_opp, contract["side_aligned_ridge_cuts"])
    rows = []
    for s in range(4):
        mask = opp_idx == s
        n = int(mask.sum())
        if n < MIN_STRATUM_ROWS:
            raise RuntimeError(f"AXB opportunity stratum {s+1} has only {n} rows")
        cfg = contract["strata"][s]
        rp = risk_pred[mask]; op = side_opp[mask]; rz = realized[mask]
        low = rp < float(cfg["predicted_mae_z_q25"])
        high = rp >= float(cfg["predicted_mae_z_q75"])
        if int(low.sum()) < MIN_TAIL_ROWS or int(high.sum()) < MIN_TAIL_ROWS:
            raise RuntimeError(f"AXB opportunity stratum {s+1} sparse frozen tails low={int(low.sum())} high={int(high.sum())}")
        low_mean = float(np.mean(rz[low])); high_mean = float(np.mean(rz[high]))
        rows.append({
            "stratum": s + 1,
            "rows": n,
            "side_aligned_ridge_mean": float(np.mean(op)),
            "side_aligned_ridge_median": float(np.median(op)),
            "risk_vs_realized_pearson": _corr(rp, rz),
            "risk_vs_realized_spearman": _spearman(rp, rz),
            "opportunity_vs_realized_mae_spearman": _spearman(op, rz),
            "risk_vs_opportunity_spearman": _spearman(rp, op),
            "frozen_low_risk_rows": int(low.sum()),
            "frozen_high_risk_rows": int(high.sum()),
            "frozen_low_risk_realized_mae_z_mean": low_mean,
            "frozen_high_risk_realized_mae_z_mean": high_mean,
            "high_minus_low_realized_mae_z_mean": float(high_mean - low_mean),
        })
    rank = np.asarray([r["risk_vs_realized_spearman"] for r in rows], dtype=float)
    sep = np.asarray([r["high_minus_low_realized_mae_z_mean"] for r in rows], dtype=float)
    return {
        "strata": rows,
        "positive_risk_spearman_strata": int(np.sum(rank > 0)),
        "positive_high_minus_low_strata": int(np.sum(sep > 0)),
        "median_risk_spearman": float(np.median(rank)),
        "minimum_risk_spearman": float(np.min(rank)),
        "median_high_minus_low_realized_mae_z_mean": float(np.median(sep)),
        "minimum_high_minus_low_realized_mae_z_mean": float(np.min(sep)),
        "diagnostic_gate": {
            "pass": bool(np.sum(rank > 0) >= MIN_POSITIVE_STRATA and np.sum(sep > 0) >= MIN_POSITIVE_STRATA),
            "minimum_positive_strata_of_4": MIN_POSITIVE_STRATA,
            "checks": {
                "positive_risk_spearman_strata_ge_3": bool(np.sum(rank > 0) >= MIN_POSITIVE_STRATA),
                "positive_frozen_high_minus_low_strata_ge_3": bool(np.sum(sep > 0) >= MIN_POSITIVE_STRATA),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--axb-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    features = list(BASE_FEATURES)
    deep_raw = load_deep(args.deep_root)
    deep = _add_move_target(_prepare(deep_bars(stitch_deep(deep_raw, deep_roll_schedule(deep_raw)))))
    axb = _add_move_target(_prepare(load_axb_mnq(args.axb_root)))

    pre = deep[(deep["timestamp"] < CUTOFF) & deep["target_move_z"].notna() & deep["long_mae_z"].notna() & deep["short_mae_z"].notna()].copy()
    train = pre.iloc[:-HORIZON].copy()
    test = axb[(axb["timestamp"] >= CUTOFF) & axb["target_move_z"].notna() & axb["long_mae_z"].notna() & axb["short_mae_z"].notna()].copy()
    if len(train) < 50000 or len(test) < MIN_TEST_ROWS:
        raise RuntimeError(f"insufficient train/test rows train={len(train)} test={len(test)}")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("chronology overlap")

    x_train = train[features].to_numpy(float); x_test = test[features].to_numpy(float)
    move_train = train["target_move_z"].to_numpy(float)
    sides = {}
    for side, target, direction in (("long", "long_mae_z", 1.0), ("short", "short_mae_z", -1.0)):
        risk_train = train[target].to_numpy(float)
        oof_risk, oof_move, oof_realized, folds = _oof_predictions(x_train, risk_train, move_train)
        oof_side_opp = direction * oof_move
        contract = _training_contract(oof_risk, oof_side_opp, oof_realized)
        rm = risk_model(); rm.fit(x_train, risk_train)
        em = ridge_model(); em.fit(x_train, move_train)
        risk_pred = np.asarray(rm.predict(x_test), dtype=float)
        side_opp = direction * np.asarray(em.predict(x_test), dtype=float)
        realized = test[target].to_numpy(float)
        sides[side] = {
            "oof_folds": folds,
            "training_oof_contract": contract,
            "axb_full": {
                "rows": int(len(test)),
                "risk_vs_realized_pearson": _corr(risk_pred, realized),
                "risk_vs_realized_spearman": _spearman(risk_pred, realized),
                "opportunity_vs_realized_mae_spearman": _spearman(side_opp, realized),
                "risk_vs_opportunity_spearman": _spearman(risk_pred, side_opp),
            },
            "axb_within_frozen_opportunity_strata": _test_strata(risk_pred, side_opp, realized, contract),
        }
        print(side, json.dumps(sides[side], sort_keys=True))

    result = {
        "schema": "foundry.mnq_h24_mae_opportunity_strata.v1",
        "research_only": True,
        "promotion_authority": False,
        "policy_authority": False,
        "training_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176 strictly pre-2026",
        "test_source": f"axb0306/cme-futures-ohlc@{AXB_PIN}",
        "deep_timestamp_contract": contract_receipt(),
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "risk_model": "same fixed H24 quantile=0.80 MAE model family",
        "opportunity_model": "same Ridge(alpha=10) normalized H24 expected-move model family; side-aligned score = ridge score for long and negative ridge score for short",
        "question": "does MAE retain adverse-excursion rank/separation information after conditioning on a frozen side-aligned Ridge expected-move/opportunity state?",
        "contract": "derive side-aligned Ridge quartile cuts and conditional MAE q25/q75 cuts only from chronological pre-2026 OOF rows; apply unchanged to all four AXB strata; report every stratum; no selection",
        "diagnostic_gate": "at least 3/4 frozen opportunity strata must have positive MAE predicted-vs-realized Spearman and positive frozen high-risk minus low-risk realized normalized MAE; diagnostic only because AXB is consumed",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sides": sides,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_MAE_OPPORTUNITY_STRATA=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
