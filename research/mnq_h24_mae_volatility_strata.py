from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.expanded_regime_ablation import BASE_FEATURES
from research.mnq_expected_move_axb_2026 import AXB_PIN, CUTOFF, HORIZON, load_axb_mnq
from research.mnq_external_transfer_validation import deep_bars, deep_roll_schedule, stitch_deep
from research.mnq_h24_mae_axb_transfer import _corr, _prepare, _spearman
from research.mnq_h24_mae_risk_specialist import risk_model

MIN_TEST_ROWS = 3000
MIN_STRATUM_ROWS = 700
MIN_TAIL_ROWS = 50
VOL_QUANTILES = (0.25, 0.50, 0.75)
RISK_TAIL_QUANTILES = (0.25, 0.75)
MIN_POSITIVE_STRATA = 3


def _oof_predictions(
    x: np.ndarray, y: np.ndarray, rv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    rv_rows: list[np.ndarray] = []
    folds: list[dict] = []
    for i in range(4):
        test_start = first + i * fold
        test_end = n if i == 3 else first + (i + 1) * fold
        train_end = test_start - HORIZON
        if train_end < 5000 or test_end - test_start < 1000:
            raise RuntimeError(
                f"invalid OOF fold {i}: train={train_end} test={test_end-test_start}"
            )
        model = risk_model()
        model.fit(x[:train_end], y[:train_end])
        preds.append(np.asarray(model.predict(x[test_start:test_end]), dtype=float))
        truth.append(np.asarray(y[test_start:test_end], dtype=float))
        rv_rows.append(np.asarray(rv[test_start:test_end], dtype=float))
        folds.append(
            {
                "fold": i,
                "train_rows": int(train_end),
                "test_rows": int(test_end - test_start),
            }
        )
    return (
        np.concatenate(preds),
        np.concatenate(truth),
        np.concatenate(rv_rows),
        folds,
    )


def _stratum_index(values: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), np.asarray(cuts, dtype=float), right=False)


def _training_contract(
    pred: np.ndarray, realized: np.ndarray, rv: np.ndarray
) -> dict:
    vol_cuts = np.quantile(rv, VOL_QUANTILES).astype(float)
    if not np.all(np.diff(vol_cuts) > 0):
        raise RuntimeError(f"non-distinct OOF volatility cuts: {vol_cuts.tolist()}")
    vol_idx = _stratum_index(rv, vol_cuts.tolist())
    strata = []
    for stratum in range(4):
        mask = vol_idx == stratum
        if int(mask.sum()) < MIN_STRATUM_ROWS:
            raise RuntimeError(
                f"OOF volatility stratum {stratum + 1} has only {int(mask.sum())} rows"
            )
        risk_cuts = np.quantile(pred[mask], RISK_TAIL_QUANTILES).astype(float)
        if not np.all(np.diff(risk_cuts) > 0):
            raise RuntimeError(
                f"non-distinct risk cuts in OOF volatility stratum {stratum + 1}: "
                f"{risk_cuts.tolist()}"
            )
        strata.append(
            {
                "stratum": stratum + 1,
                "oof_rows": int(mask.sum()),
                "rv_120_min": None if stratum == 0 else float(vol_cuts[stratum - 1]),
                "rv_120_max_exclusive": None if stratum == 3 else float(vol_cuts[stratum]),
                "predicted_mae_z_q25": float(risk_cuts[0]),
                "predicted_mae_z_q75": float(risk_cuts[1]),
                "oof_predicted_vs_realized_pearson": _corr(pred[mask], realized[mask]),
                "oof_predicted_vs_realized_spearman": _spearman(pred[mask], realized[mask]),
                "oof_rv_vs_realized_spearman": _spearman(rv[mask], realized[mask]),
            }
        )
    return {
        "volatility_quantiles": list(VOL_QUANTILES),
        "rv_120_cuts": [float(x) for x in vol_cuts],
        "conditional_risk_tail_quantiles": list(RISK_TAIL_QUANTILES),
        "strata": strata,
    }


def _test_strata(
    pred: np.ndarray,
    realized: np.ndarray,
    rv: np.ndarray,
    contract: dict,
) -> dict:
    vol_idx = _stratum_index(rv, contract["rv_120_cuts"])
    rows = []
    for stratum in range(4):
        mask = vol_idx == stratum
        n = int(mask.sum())
        if n < MIN_STRATUM_ROWS:
            raise RuntimeError(f"AX B volatility stratum {stratum + 1} has only {n} rows")
        cfg = contract["strata"][stratum]
        p = pred[mask]
        r = realized[mask]
        v = rv[mask]
        low = p < float(cfg["predicted_mae_z_q25"])
        high = p >= float(cfg["predicted_mae_z_q75"])
        if int(low.sum()) < MIN_TAIL_ROWS or int(high.sum()) < MIN_TAIL_ROWS:
            raise RuntimeError(
                f"AXB stratum {stratum + 1} sparse frozen tails "
                f"low={int(low.sum())} high={int(high.sum())}"
            )
        q1_mean = float(np.mean(r[low]))
        q4_mean = float(np.mean(r[high]))
        rows.append(
            {
                "stratum": stratum + 1,
                "rows": n,
                "rv_120_mean": float(np.mean(v)),
                "rv_120_median": float(np.median(v)),
                "predicted_vs_realized_pearson": _corr(p, r),
                "predicted_vs_realized_spearman": _spearman(p, r),
                "rv_vs_realized_spearman": _spearman(v, r),
                "frozen_low_risk_rows": int(low.sum()),
                "frozen_high_risk_rows": int(high.sum()),
                "frozen_low_risk_realized_mae_z_mean": q1_mean,
                "frozen_high_risk_realized_mae_z_mean": q4_mean,
                "high_minus_low_realized_mae_z_mean": float(q4_mean - q1_mean),
            }
        )
    spearman = np.asarray(
        [row["predicted_vs_realized_spearman"] for row in rows], dtype=float
    )
    separation = np.asarray(
        [row["high_minus_low_realized_mae_z_mean"] for row in rows], dtype=float
    )
    risk_minus_rv = np.asarray(
        [
            abs(float(row["predicted_vs_realized_spearman"]))
            - abs(float(row["rv_vs_realized_spearman"]))
            for row in rows
        ],
        dtype=float,
    )
    return {
        "strata": rows,
        "positive_spearman_strata": int(np.sum(spearman > 0)),
        "positive_high_minus_low_strata": int(np.sum(separation > 0)),
        "median_spearman": float(np.median(spearman)),
        "minimum_spearman": float(np.min(spearman)),
        "median_high_minus_low_realized_mae_z_mean": float(np.median(separation)),
        "minimum_high_minus_low_realized_mae_z_mean": float(np.min(separation)),
        "median_abs_spearman_advantage_over_rv": float(np.median(risk_minus_rv)),
        "diagnostic_gate": {
            "pass": bool(
                np.sum(spearman > 0) >= MIN_POSITIVE_STRATA
                and np.sum(separation > 0) >= MIN_POSITIVE_STRATA
            ),
            "minimum_positive_strata_of_4": MIN_POSITIVE_STRATA,
            "checks": {
                "positive_predicted_vs_realized_spearman_strata_ge_3": bool(
                    np.sum(spearman > 0) >= MIN_POSITIVE_STRATA
                ),
                "positive_frozen_high_minus_low_strata_ge_3": bool(
                    np.sum(separation > 0) >= MIN_POSITIVE_STRATA
                ),
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
    deep = _prepare(deep_bars(stitch_deep(deep_raw, deep_roll_schedule(deep_raw))))
    axb = _prepare(load_axb_mnq(args.axb_root))

    pre = deep[
        (deep["timestamp"] < CUTOFF)
        & deep["long_mae_z"].notna()
        & deep["short_mae_z"].notna()
    ].copy()
    if len(pre) <= HORIZON:
        raise RuntimeError("insufficient pre-2026 deep rows")
    train = pre.iloc[:-HORIZON].copy()
    test = axb[
        (axb["timestamp"] >= CUTOFF)
        & axb["long_mae_z"].notna()
        & axb["short_mae_z"].notna()
    ].copy()
    if len(train) < 50000 or len(test) < MIN_TEST_ROWS:
        raise RuntimeError(f"insufficient train/test rows train={len(train)} test={len(test)}")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("chronology overlap")

    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    rv_train = train["rv_120"].to_numpy(float)
    rv_test = test["rv_120"].to_numpy(float)

    sides = {}
    for side, target in (("long", "long_mae_z"), ("short", "short_mae_z")):
        y_train = train[target].to_numpy(float)
        oof_pred, oof_realized, oof_rv, folds = _oof_predictions(
            x_train, y_train, rv_train
        )
        contract = _training_contract(oof_pred, oof_realized, oof_rv)
        model = risk_model()
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test), dtype=float)
        realized = test[target].to_numpy(float)
        sides[side] = {
            "oof_folds": folds,
            "training_oof_contract": contract,
            "axb_full": {
                "rows": int(len(test)),
                "predicted_vs_realized_pearson": _corr(pred, realized),
                "predicted_vs_realized_spearman": _spearman(pred, realized),
                "rv_vs_realized_spearman": _spearman(rv_test, realized),
            },
            "axb_within_frozen_volatility_strata": _test_strata(
                pred, realized, rv_test, contract
            ),
        }
        print(side, json.dumps(sides[side], sort_keys=True))

    result = {
        "schema": "foundry.mnq_h24_mae_volatility_strata.v1",
        "research_only": True,
        "promotion_authority": False,
        "policy_authority": False,
        "training_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176 strictly pre-2026",
        "test_source": f"axb0306/cme-futures-ohlc@{AXB_PIN}",
        "deep_timestamp_contract": contract_receipt(),
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "risk_model": "same fixed HistGradientBoostingRegressor quantile=0.80 family as corrected H24 MAE transfer; separate long/short models",
        "question": "does predicted normalized H24 MAE retain rank/separation information within volatility strata frozen entirely from pre-2026 OOF evidence, rather than merely restating rv_120 level?",
        "contract": "derive rv_120 quartile cuts and conditional predicted-MAE q25/q75 cuts only from chronological pre-2026 OOF rows; apply unchanged to all four AXB strata; report every stratum with no selection; normalized MAE target already divides by close*rv_120*sqrt(H24)",
        "diagnostic_gate": "at least 3/4 frozen volatility strata must have positive predicted-vs-realized Spearman and positive frozen high-risk minus low-risk realized normalized MAE; diagnostic only because AXB is already consumed",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sides": sides,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_MAE_VOLATILITY_STRATA=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
