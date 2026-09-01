from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.expanded_regime_ablation import BASE_FEATURES
from research.mnq_external_transfer_validation import deep_bars, deep_roll_schedule, stitch_deep
from research.mnq_h24_mae_axb_transfer import _corr, _prepare, _spearman
from research.mnq_h24_mae_risk_specialist import risk_model

SYMBOL = "MNQ=F"
SOURCE_START = "2026-07-06"
SOURCE_END = "2026-08-29"  # end-exclusive; last complete Friday before this research run
HORIZON = 24
MIN_TEST_ROWS = 2500
MIN_MEDIAN_DAILY_12M_BARS = 80
BIN_QUANTILES = (0.25, 0.50, 0.75)
TAIL_QUANTILE = 0.80
MIN_BIN_ROWS = 50
MIN_PHASE_ROWS = 40
PREDECLARED_MIN_FULL_SPEARMAN = 0.15
PREDECLARED_MIN_BIN_MEAN_RANK_SPEARMAN = 0.80
PREDECLARED_MIN_Q4_MINUS_Q1_TAIL_RATE = 0.10
PREDECLARED_MIN_POSITIVE_PHASES = 18


def _normalize_yahoo_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("Yahoo returned no MNQ intraday rows")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if SYMBOL in frame.columns.get_level_values(-1):
            frame = frame.xs(SYMBOL, axis=1, level=-1)
        elif SYMBOL in frame.columns.get_level_values(0):
            frame = frame.xs(SYMBOL, axis=1, level=0)
        else:
            raise RuntimeError(f"unexpected Yahoo MultiIndex columns {frame.columns.tolist()[:8]}")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Yahoo missing required columns {missing}; got {list(frame.columns)}")
    if getattr(frame.index, "tz", None) is None:
        raise RuntimeError("Yahoo intraday index unexpectedly timezone-naive")
    frame = frame[required].copy()
    frame["timestamp"] = pd.to_datetime(frame.index, utc=True, errors="raise")
    for c in required:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep=False).reset_index(drop=True)
    if frame.empty or not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Yahoo normalized timestamps invalid")
    return frame[["timestamp", *required]]


def load_yahoo_mnq() -> tuple[pd.DataFrame, dict]:
    raw = yf.download(
        SYMBOL,
        start=SOURCE_START,
        end=SOURCE_END,
        interval="2m",
        auto_adjust=False,
        prepost=True,
        progress=False,
        threads=False,
    )
    source = _normalize_yahoo_columns(raw)
    canonical = source.copy()
    canonical["timestamp"] = canonical["timestamp"].map(lambda x: x.isoformat())
    source_bytes = canonical.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    w = source.set_index("timestamp")
    bars = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        observed_2m=("close", "count"),
    )
    bars = bars[bars["observed_2m"] > 0].reset_index()
    bars["observed_minutes"] = bars["observed_2m"].astype(float) * 2.0
    if bars.empty:
        raise RuntimeError("Yahoo produced no 12-minute bars")
    daily = bars.groupby(bars["timestamp"].dt.date).size().astype(float)
    median_daily = float(daily.median())
    fullish_fraction = float(np.mean(bars["observed_2m"].to_numpy(float) >= 5.0))
    if len(bars) < MIN_TEST_ROWS:
        raise RuntimeError(f"Yahoo 12-minute history too short: {len(bars)} rows")
    if median_daily < MIN_MEDIAN_DAILY_12M_BARS:
        raise RuntimeError(
            f"Yahoo session coverage too shallow: median daily 12m bars={median_daily:.1f} "
            f"< {MIN_MEDIAN_DAILY_12M_BARS}"
        )
    receipt = {
        "provider": "Yahoo Finance via yfinance",
        "provider_symbol": SYMBOL,
        "requested_start": SOURCE_START,
        "requested_end_exclusive": SOURCE_END,
        "requested_interval": "2m",
        "prepost": True,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "raw_rows": int(len(source)),
        "raw_first_timestamp": source["timestamp"].min().isoformat(),
        "raw_last_timestamp": source["timestamp"].max().isoformat(),
        "raw_canonical_csv_sha256": source_sha256,
        "bars_12m": int(len(bars)),
        "median_daily_12m_bars": median_daily,
        "fraction_12m_bars_with_at_least_5_of_6_2m_samples": fullish_fraction,
        "minute_density_contract": "observed_minutes = observed_2m_samples * 2, clipped by shared feature code at 12 minutes",
        "bar_contract": "provider intraday timestamp normalized to UTC bar start; 2m OHLCV resampled to UTC-origin 12m left-closed/left-labeled bars",
        "raw_data_redistributed": False,
    }
    return bars, receipt


def _oof_predictions(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    n = len(y)
    first = n // 2
    fold = (n - first) // 4
    preds: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    folds: list[dict] = []
    for i in range(4):
        ts = first + i * fold
        te = n if i == 3 else first + (i + 1) * fold
        train_end = ts - HORIZON
        if train_end < 5000 or te - ts < 1000:
            raise RuntimeError(f"invalid OOF fold {i}: train={train_end} test={te-ts}")
        model = risk_model()
        model.fit(x[:train_end], y[:train_end])
        preds.append(np.asarray(model.predict(x[ts:te]), dtype=float))
        truth.append(np.asarray(y[ts:te], dtype=float))
        folds.append({"fold": i, "train_rows": int(train_end), "test_rows": int(te-ts)})
    return np.concatenate(preds), np.concatenate(truth), folds


def _fixed_thresholds(oof_pred: np.ndarray, oof_realized: np.ndarray) -> dict:
    cuts = np.quantile(oof_pred, BIN_QUANTILES).astype(float)
    if not np.all(np.diff(cuts) > 0):
        raise RuntimeError(f"non-distinct training OOF risk-bin cuts {cuts.tolist()}")
    return {
        "predicted_mae_z_bin_quantiles": list(BIN_QUANTILES),
        "predicted_mae_z_bin_cuts": [float(x) for x in cuts],
        "realized_mae_z_tail_quantile": TAIL_QUANTILE,
        "realized_mae_z_tail_threshold": float(np.quantile(oof_realized, TAIL_QUANTILE)),
        "oof_rows": int(len(oof_pred)),
        "oof_pearson": _corr(oof_pred, oof_realized),
        "oof_spearman": _spearman(oof_pred, oof_realized),
    }


def _bin_index(pred: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.digitize(np.asarray(pred, dtype=float), np.asarray(cuts, dtype=float), right=False)


def _bin_receipt(pred: np.ndarray, realized: np.ndarray, cuts: list[float], tail_threshold: float) -> dict:
    pred = np.asarray(pred, dtype=float)
    realized = np.asarray(realized, dtype=float)
    bins = _bin_index(pred, cuts)
    rows = []
    means = []
    tail_rates = []
    for b in range(4):
        mask = bins == b
        n = int(mask.sum())
        if n < MIN_BIN_ROWS:
            raise RuntimeError(f"test risk bin {b + 1} has only {n} rows")
        rv = realized[mask]
        pv = pred[mask]
        mean_realized = float(np.mean(rv))
        tail_rate = float(np.mean(rv >= tail_threshold))
        means.append(mean_realized)
        tail_rates.append(tail_rate)
        rows.append(
            {
                "bin": b + 1,
                "rows": n,
                "predicted_mae_z_mean": float(np.mean(pv)),
                "predicted_mae_z_median": float(np.median(pv)),
                "realized_mae_z_mean": mean_realized,
                "realized_mae_z_median": float(np.median(rv)),
                "realized_mae_z_p80": float(np.quantile(rv, 0.80)),
                "training_oof_tail_exceedance_rate": tail_rate,
            }
        )
    rank_spearman = float(pd.Series([1.0, 2.0, 3.0, 4.0]).corr(pd.Series(means), method="spearman"))
    return {
        "bins": rows,
        "realized_mean_rank_spearman": rank_spearman,
        "strictly_increasing_realized_means": bool(np.all(np.diff(np.asarray(means)) > 0)),
        "q4_minus_q1_realized_mean": float(means[3] - means[0]),
        "q4_minus_q1_tail_exceedance_rate": float(tail_rates[3] - tail_rates[0]),
    }


def _phase_receipt(pred: np.ndarray, realized: np.ndarray, cuts: list[float]) -> dict:
    bins = _bin_index(pred, cuts)
    rows = []
    for phase in range(HORIZON):
        idx = np.arange(phase, len(pred), HORIZON, dtype=int)
        if len(idx) < MIN_PHASE_ROWS:
            continue
        low = idx[bins[idx] == 0]
        high = idx[bins[idx] == 3]
        high_low = None
        if len(low) >= 5 and len(high) >= 5:
            high_low = float(np.mean(realized[high]) - np.mean(realized[low]))
        rows.append(
            {
                "phase": phase,
                "rows": int(len(idx)),
                "pearson": _corr(pred[idx], realized[idx]),
                "spearman": _spearman(pred[idx], realized[idx]),
                "q4_minus_q1_realized_mean": high_low,
                "q1_rows": int(len(low)),
                "q4_rows": int(len(high)),
            }
        )
    if len(rows) != HORIZON:
        raise RuntimeError(f"expected {HORIZON} usable non-overlap phases, got {len(rows)}")
    spearman = np.asarray([r["spearman"] for r in rows if r["spearman"] is not None], dtype=float)
    high_low = np.asarray(
        [r["q4_minus_q1_realized_mean"] for r in rows if r["q4_minus_q1_realized_mean"] is not None],
        dtype=float,
    )
    return {
        "phases": rows,
        "positive_spearman_phases": int(np.sum(spearman > 0)),
        "spearman_phase_count": int(len(spearman)),
        "median_phase_spearman": float(np.median(spearman)),
        "minimum_phase_spearman": float(np.min(spearman)),
        "positive_q4_minus_q1_phases": int(np.sum(high_low > 0)),
        "q4_minus_q1_phase_count": int(len(high_low)),
        "median_phase_q4_minus_q1_realized_mean": float(np.median(high_low)),
        "minimum_phase_q4_minus_q1_realized_mean": float(np.min(high_low)),
    }


def _monthly(test: pd.DataFrame, pred: np.ndarray, realized: np.ndarray, cuts: list[float]) -> list[dict]:
    work = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "pred": np.asarray(pred, dtype=float),
            "realized": np.asarray(realized, dtype=float),
        }
    )
    work["bin"] = _bin_index(work["pred"].to_numpy(float), cuts)
    work["month"] = pd.to_datetime(work["timestamp"], utc=True).dt.to_period("M").astype(str)
    out = []
    for month, g in work.groupby("month", sort=True):
        if len(g) < 500:
            continue
        low = g[g["bin"] == 0]["realized"].to_numpy(float)
        high = g[g["bin"] == 3]["realized"].to_numpy(float)
        out.append(
            {
                "month": month,
                "rows": int(len(g)),
                "pearson": _corr(g["pred"].to_numpy(float), g["realized"].to_numpy(float)),
                "spearman": _spearman(g["pred"].to_numpy(float), g["realized"].to_numpy(float)),
                "q4_minus_q1_realized_mean": float(np.mean(high) - np.mean(low)) if len(low) >= 20 and len(high) >= 20 else None,
                "q1_rows": int(len(low)),
                "q4_rows": int(len(high)),
            }
        )
    return out


def _predeclared_gate(full_spearman: float | None, bins: dict, phases: dict) -> dict:
    checks = {
        "full_spearman_ge_0p15": full_spearman is not None and full_spearman >= PREDECLARED_MIN_FULL_SPEARMAN,
        "bin_mean_rank_spearman_ge_0p80": bins["realized_mean_rank_spearman"] >= PREDECLARED_MIN_BIN_MEAN_RANK_SPEARMAN,
        "q4_realized_mean_gt_q1": bins["q4_minus_q1_realized_mean"] > 0,
        "q4_minus_q1_tail_rate_ge_0p10": bins["q4_minus_q1_tail_exceedance_rate"] >= PREDECLARED_MIN_Q4_MINUS_Q1_TAIL_RATE,
        "positive_phase_spearman_ge_18_of_24": phases["positive_spearman_phases"] >= PREDECLARED_MIN_POSITIVE_PHASES,
        "positive_phase_q4_minus_q1_ge_18_of_24": phases["positive_q4_minus_q1_phases"] >= PREDECLARED_MIN_POSITIVE_PHASES,
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    features = list(BASE_FEATURES)
    deep_raw = load_deep(args.deep_root)
    deep = _prepare(deep_bars(stitch_deep(deep_raw, deep_roll_schedule(deep_raw))))
    pre = deep[(deep["timestamp"] < pd.Timestamp("2026-01-01", tz="UTC")) & deep["long_mae_z"].notna() & deep["short_mae_z"].notna()].copy()
    if len(pre) <= HORIZON:
        raise RuntimeError("insufficient corrected pre-2026 deep rows")
    train = pre.iloc[:-HORIZON].copy()
    if len(train) < 50000:
        raise RuntimeError(f"insufficient training rows {len(train)}")

    yahoo_bars, source_receipt = load_yahoo_mnq()
    test = _prepare(yahoo_bars)
    test = test[test["long_mae_z"].notna() & test["short_mae_z"].notna()].copy().reset_index(drop=True)
    if len(test) < MIN_TEST_ROWS:
        raise RuntimeError(f"insufficient post-feature Yahoo test rows {len(test)}")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("training/test chronology overlap")
    if test["timestamp"].min() <= pd.Timestamp("2026-04-15", tz="UTC"):
        raise RuntimeError("test block is not strictly post-AXB")

    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    sides = {}
    for side, target in (("long", "long_mae_z"), ("short", "short_mae_z")):
        y_train = train[target].to_numpy(float)
        oof_pred, oof_realized, folds = _oof_predictions(x_train, y_train)
        thresholds = _fixed_thresholds(oof_pred, oof_realized)
        model = risk_model()
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test), dtype=float)
        realized = test[target].to_numpy(float)
        cuts = thresholds["predicted_mae_z_bin_cuts"]
        tail = thresholds["realized_mae_z_tail_threshold"]
        full_pearson = _corr(pred, realized)
        full_spearman = _spearman(pred, realized)
        bins = _bin_receipt(pred, realized, cuts, tail)
        phases = _phase_receipt(pred, realized, cuts)
        sides[side] = {
            "oof_folds": folds,
            "training_oof_thresholds": thresholds,
            "test_rows": int(len(test)),
            "full_pearson": full_pearson,
            "full_spearman": full_spearman,
            "risk_stratification": bins,
            "nonoverlap_h24_phase_robustness": phases,
            "monthly": _monthly(test, pred, realized, cuts),
            "predeclared_statistical_gate": _predeclared_gate(full_spearman, bins, phases),
        }
        print(side, json.dumps(sides[side], sort_keys=True))

    result = {
        "schema": "foundry.mnq_h24_mae_yahoo_julaug_transfer.v1",
        "research_only": True,
        "promotion_authority": False,
        "policy_authority": False,
        "training_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176 strictly pre-2026",
        "deep_timestamp_contract": contract_receipt(),
        "test_source": source_receipt,
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "risk_model": "same frozen HistGradientBoostingRegressor quantile=0.80 family as corrected H24 MAE research; separate long/short models",
        "contract": "untouched post-AXB Jul-Aug 2026 external-source transfer; risk-bin cuts and realized-tail threshold are learned only from chronological pre-2026 OOF predictions/outcomes; no test-set threshold selection, veto, PnL optimization, sizing, capital routing, or runtime authority",
        "predeclared_gate": {
            "full_spearman_minimum": PREDECLARED_MIN_FULL_SPEARMAN,
            "bin_mean_rank_spearman_minimum": PREDECLARED_MIN_BIN_MEAN_RANK_SPEARMAN,
            "q4_minus_q1_tail_rate_minimum": PREDECLARED_MIN_Q4_MINUS_Q1_TAIL_RATE,
            "minimum_positive_nonoverlap_phases_of_24": PREDECLARED_MIN_POSITIVE_PHASES,
        },
        "train_rows": int(len(train)),
        "train_last_timestamp": train["timestamp"].max().isoformat(),
        "test_rows": int(len(test)),
        "test_first_timestamp": test["timestamp"].min().isoformat(),
        "test_last_timestamp": test["timestamp"].max().isoformat(),
        "sides": sides,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_MAE_YAHOO_JULAUG_TRANSFER=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
