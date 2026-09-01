from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.deep_mnq_source_contract import contract_receipt, load_deep
from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_expected_move_regression import chronological_oof_threshold, make_regressor
from research.mnq_external_transfer_validation import deep_bars, deep_roll_schedule, stitch_deep
from research.mnq_model_family_weekly_challenge import trade_week_key
from research.nq_to_mnq_execution_transfer import POINT_COSTS, phase_audit

SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
RISK_PREREQ_COMMIT = "d2e1e0c4fd80235a76b0969a249f90eeb7b227d1"
DISCOVERY_START = pd.Timestamp("2023-07-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2026-01-01", tz="UTC")
RIDGE_HORIZON = 12
RISK_HORIZON = 4
RISK_QUANTILE = 0.80
MIN_PAIRED_WEEKS = 80
RISK_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "range_frac", "body_frac", "log_volume", "volume_z_24", "volume_z_120",
    "rv_12", "rv_60", "rv_120", "ema12_dist", "ema48_dist",
    "mom_24", "mom_120", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
RISK_TARGETS = ("long_mae_z", "short_mae_z")


def verify_prerequisite(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("timestamp_correction_generation") != "timefixed_r1":
        raise RuntimeError("1H risk prerequisite is not timestamp-corrected")
    contract = data.get("deep_timestamp_contract", {})
    expected = {
        "source_timestamp_timezone": "America/New_York",
        "source_bar_label": "right_close",
        "normalized_timestamp": "UTC bar start",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"1H risk prerequisite timestamp contract mismatch: {key}")
    decisions = data.get("decisions", {}).get(str(RISK_HORIZON), {})
    for target in ("long_mae_z", "short_mae_z", "future_rv_z"):
        if decisions.get(target, {}).get("action") != "advance_to_separate_economic_consumer":
            raise RuntimeError(f"1H risk prerequisite failed H{RISK_HORIZON}/{target}")
    return {
        "receipt_sha256": data.get("receipt_sha256"),
        "timestamp_correction_generation": data.get("timestamp_correction_generation"),
        "deep_timestamp_contract": contract,
        "h4_decisions": decisions,
    }


def bars_1h(stitched: pd.DataFrame) -> pd.DataFrame:
    w = stitched.set_index("timestamp")
    bars = w.resample("60min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), observed_minutes=("close", "count"),
        source_contract=("symbol", "first"), source_contract_last=("symbol", "last"),
    )
    bars = bars[bars["observed_minutes"] > 0].copy()
    mixed = bars["source_contract"] != bars["source_contract_last"]
    bars = bars.loc[~mixed].drop(columns=["source_contract_last"]).reset_index()
    bars.attrs["dropped_roll_boundary_bars"] = int(mixed.sum())
    return bars


def _future_extreme(series: pd.Series, horizon: int, kind: str) -> pd.Series:
    shifted = pd.concat([series.shift(-i) for i in range(1, horizon + 1)], axis=1)
    if kind == "max":
        return shifted.max(axis=1, skipna=False)
    if kind == "min":
        return shifted.min(axis=1, skipna=False)
    raise ValueError(kind)


def risk_segment(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy().reset_index(drop=True)
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    open_ = g["open"].astype(float)
    log_close = np.log(close)
    ret1 = log_close.diff()
    for lag in (1, 3, 6, 12):
        g[f"ret_{lag}"] = log_close.diff(lag)
    g["range_frac"] = (high - low) / close.replace(0, np.nan)
    g["body_frac"] = (close - open_) / open_.replace(0, np.nan)
    g["log_volume"] = np.log1p(g["volume"].astype(float).clip(lower=0))
    for window in (24, 120):
        mean = g["log_volume"].rolling(window, min_periods=window).mean()
        std = g["log_volume"].rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
        g[f"volume_z_{window}"] = (g["log_volume"] - mean) / std
    for window in (12, 60, 120):
        g[f"rv_{window}"] = ret1.rolling(window, min_periods=window).std(ddof=0)
    for span in (12, 48):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        g[f"ema{span}_dist"] = close / ema - 1.0
    g["mom_24"] = close.pct_change(24)
    g["mom_120"] = close.pct_change(120)
    hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
    g["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    g["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    dow = g["timestamp"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * math.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * math.pi * dow / 7.0)
    future_high = _future_extreme(high, RISK_HORIZON, "max")
    future_low = _future_extreme(low, RISK_HORIZON, "min")
    scale = close * g["rv_120"] * math.sqrt(RISK_HORIZON)
    g["long_mae_z"] = (close - future_low).clip(lower=0) / scale.replace(0, np.nan)
    g["short_mae_z"] = (future_high - close).clip(lower=0) / scale.replace(0, np.nan)
    g["available_at"] = g["timestamp"] + pd.Timedelta(hours=1)
    return g


def build_risk_matrix(bars: pd.DataFrame) -> pd.DataFrame:
    work = bars.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    segment = work["source_contract"].ne(work["source_contract"].shift()).cumsum()
    out = pd.concat([risk_segment(g) for _, g in work.groupby(segment, sort=False)], ignore_index=True)
    keep = ["timestamp", "available_at", "source_contract", *RISK_FEATURES, *RISK_TARGETS]
    out = out[keep].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=RISK_FEATURES).reset_index(drop=True)
    if out["timestamp"].duplicated().any() or not out["timestamp"].is_monotonic_increasing:
        raise RuntimeError("1H risk matrix timestamps are not unique/increasing")
    return out


def risk_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=RISK_QUANTILE, learning_rate=0.05, max_iter=180,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=42,
    )


def opportunity_matrix(stitched: pd.DataFrame) -> pd.DataFrame:
    features = list(BASE_FEATURES)
    work = _add_features(deep_bars(stitched))
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    work = work[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    work["point_move"] = work["close"].shift(-RIDGE_HORIZON) - work["close"]
    scale = work["close"].astype(float) * work["rv_120"].astype(float) * math.sqrt(RIDGE_HORIZON)
    work["target_move_z"] = work["point_move"] / scale.replace(0, np.nan)
    work["trade_week"] = trade_week_key(work["timestamp"])
    return work


def fit_quarter(op: pd.DataFrame, risk: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict] | None:
    features = list(BASE_FEATURES)
    op_mask = (op["timestamp"] >= start) & (op["timestamp"] < end) & op["target_move_z"].notna() & op["point_move"].notna()
    op_idx = np.flatnonzero(op_mask.to_numpy())
    if len(op_idx) < 2000:
        return None
    op_test_start = int(op_idx[0])
    op_train_end = op_test_start - RIDGE_HORIZON
    if op_train_end < 50000:
        return None
    op_train = op.iloc[:op_train_end].copy()
    op_train = op_train[op_train["target_move_z"].notna()].copy()
    op_test = op.iloc[int(op_idx[0]): int(op_idx[-1] + 1)].copy()
    op_test = op_test[(op_test["timestamp"] >= start) & (op_test["timestamp"] < end) & op_test["target_move_z"].notna() & op_test["point_move"].notna()].copy()
    if op_train["timestamp"].max() >= op_test["timestamp"].min():
        raise RuntimeError("Ridge chronology violation")

    x_train = op_train[features].to_numpy(float)
    y_train = op_train["target_move_z"].to_numpy(float)
    ridge_threshold, ridge_oof = chronological_oof_threshold("ridge", x_train, y_train, RIDGE_HORIZON)
    ridge = make_regressor("ridge").fit(x_train, y_train)
    pred_z = ridge.predict(op_test[features].to_numpy(float))
    baseline = np.where(np.abs(pred_z) >= ridge_threshold, np.sign(pred_z), 0).astype(int)

    risk_train = risk[(risk["timestamp"] < start - pd.Timedelta(hours=RISK_HORIZON))].copy()
    risk_train = risk_train.dropna(subset=list(RISK_TARGETS))
    if len(risk_train) < 5000:
        return None
    risk_scores = risk[(risk["available_at"] <= end) & (risk["available_at"] >= start - pd.Timedelta(days=3))].copy()
    if risk_scores.empty:
        raise RuntimeError("no 1H risk scoring rows for quarter")

    rx = risk_train[RISK_FEATURES].to_numpy(float)
    sx = risk_scores[RISK_FEATURES].to_numpy(float)
    thresholds = {}
    for target in RISK_TARGETS:
        model = risk_model().fit(rx, risk_train[target].to_numpy(float))
        train_pred = model.predict(rx)
        thresholds[target] = float(np.quantile(train_pred, RISK_QUANTILE))
        risk_scores[f"pred_{target}"] = model.predict(sx)

    join_cols = ["available_at", "pred_long_mae_z", "pred_short_mae_z"]
    joined = pd.merge_asof(
        op_test.sort_values("timestamp"), risk_scores[join_cols].sort_values("available_at"),
        left_on="timestamp", right_on="available_at", direction="backward", allow_exact_matches=True,
    )
    if joined[["pred_long_mae_z", "pred_short_mae_z"]].isna().any().any():
        raise RuntimeError("missing completed-hour 1H risk score for 12Min decision")
    if (joined["available_at"] > joined["timestamp"]).any():
        raise RuntimeError("1H risk lookahead detected")

    conditioned = baseline.copy()
    long_veto = (baseline == 1) & (joined["pred_long_mae_z"].to_numpy(float) >= thresholds["long_mae_z"])
    short_veto = (baseline == -1) & (joined["pred_short_mae_z"].to_numpy(float) >= thresholds["short_mae_z"])
    conditioned[long_veto | short_veto] = 0
    joined["baseline_signal"] = baseline
    joined["conditioned_signal"] = conditioned
    joined["risk_age_minutes"] = (joined["timestamp"] - joined["available_at"]) / pd.Timedelta(minutes=1)

    receipt = {
        "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
        "ridge_train_rows": int(len(op_train)), "ridge_test_rows": int(len(op_test)),
        "ridge_train_last_timestamp": op_train["timestamp"].max().isoformat(),
        "ridge_oof": ridge_oof,
        "risk_train_rows": int(len(risk_train)),
        "risk_train_last_timestamp": risk_train["timestamp"].max().isoformat(),
        "risk_thresholds": thresholds,
        "baseline_signals": int(np.sum(baseline != 0)),
        "conditioned_signals": int(np.sum(conditioned != 0)),
        "long_signals": int(np.sum(baseline == 1)), "short_signals": int(np.sum(baseline == -1)),
        "long_vetoes": int(np.sum(long_veto)), "short_vetoes": int(np.sum(short_veto)),
        "long_veto_rate": float(np.mean(long_veto[baseline == 1])) if np.any(baseline == 1) else 0.0,
        "short_veto_rate": float(np.mean(short_veto[baseline == -1])) if np.any(baseline == -1) else 0.0,
        "risk_age_minutes": {
            "median": float(joined["risk_age_minutes"].median()),
            "p95": float(joined["risk_age_minutes"].quantile(0.95)),
            "max": float(joined["risk_age_minutes"].max()),
        },
    }
    return joined, receipt


def weekly_rows(joined: pd.DataFrame, quarter: str) -> list[dict]:
    rows = []
    for week_key, g in joined.groupby("trade_week", sort=True):
        if len(g) < 300:
            continue
        move = g["point_move"].to_numpy(float)
        base = g["baseline_signal"].to_numpy(int)
        cond = g["conditioned_signal"].to_numpy(int)
        rows.append({
            "trade_week": pd.Timestamp(week_key).isoformat(), "quarter": quarter, "rows": int(len(g)),
            "baseline": {"coverage": float(np.mean(base != 0)), "phase_audit": phase_audit(g["timestamp"], base, move, RIDGE_HORIZON)},
            "conditioned": {"coverage": float(np.mean(cond != 0)), "phase_audit": phase_audit(g["timestamp"], cond, move, RIDGE_HORIZON)},
        })
    return rows


def metric_array(rows: list[dict], policy: str, cost: float) -> tuple[np.ndarray, list[int]]:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    vals, idx = [], []
    for i, row in enumerate(rows):
        value = row[policy]["phase_audit"].get(field)
        if value is not None and np.isfinite(value):
            vals.append(float(value)); idx.append(i)
    return np.asarray(vals, dtype=float), idx


def tail_stats(arr: np.ndarray) -> dict:
    k = max(1, int(np.ceil(0.10 * len(arr))))
    sorted_arr = np.sort(arr)
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])
    draw = np.r_[0.0, cumulative] - peak
    return {
        "weeks": int(len(arr)), "median": float(np.median(arr)), "mean": float(np.mean(arr)),
        "positive_week_fraction": float(np.mean(arr > 0)), "p10": float(np.quantile(arr, 0.10)),
        "bottom10pct_mean": float(np.mean(sorted_arr[:k])), "worst_week": float(np.min(arr)),
        "cumulative_points": float(np.sum(arr)), "max_drawdown_points": float(np.min(draw)),
    }


def paired_summary(rows: list[dict], cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    base, cond, labels = [], [], []
    for row in rows:
        b = row["baseline"]["phase_audit"].get(field)
        c = row["conditioned"]["phase_audit"].get(field)
        if b is not None and c is not None and np.isfinite(b) and np.isfinite(c):
            base.append(float(b)); cond.append(float(c)); labels.append(row["trade_week"])
    b = np.asarray(base, dtype=float); c = np.asarray(cond, dtype=float); delta = c - b
    if len(delta) < MIN_PAIRED_WEEKS:
        return {"weeks": int(len(delta)), "eligible": False}
    return {
        "weeks": int(len(delta)), "eligible": True,
        "baseline": tail_stats(b), "conditioned": tail_stats(c),
        "delta": {
            "median": float(np.median(delta)), "mean": float(np.mean(delta)),
            "win_fraction": float(np.mean(delta > 0)), "tie_fraction": float(np.mean(delta == 0)),
            "p10": float(np.quantile(delta, 0.10)), "worst": float(np.min(delta)), "best": float(np.max(delta)),
        },
        "first_week": labels[0], "last_week": labels[-1],
    }


def advance_checks(summary: dict) -> dict:
    if not summary.get("eligible"):
        checks = {"paired_weeks_at_least_80": False}
    else:
        b, c, d = summary["baseline"], summary["conditioned"], summary["delta"]
        checks = {
            "paired_weeks_at_least_80": summary["weeks"] >= MIN_PAIRED_WEEKS,
            "positive_paired_median_delta": d["median"] > 0,
            "positive_paired_mean_delta": d["mean"] > 0,
            "paired_win_fraction_gt_0p50": d["win_fraction"] > 0.50,
            "conditioned_positive_week_fraction_ge_baseline": c["positive_week_fraction"] >= b["positive_week_fraction"],
            "conditioned_p10_ge_baseline": c["p10"] >= b["p10"],
            "conditioned_bottom10pct_ge_baseline": c["bottom10pct_mean"] >= b["bottom10pct_mean"],
            "conditioned_drawdown_no_worse": c["max_drawdown_points"] >= b["max_drawdown_points"],
        }
    return {"passed": all(checks.values()), "checks": checks, "failed": [k for k, v in checks.items() if not v]}


def decomposition(rows: list[dict], cost: float) -> dict:
    key = str(cost).replace(".", "p")
    field = f"median_phase_net_points_after_{key}pt"
    records = []
    for row in rows:
        b = row["baseline"]["phase_audit"].get(field); c = row["conditioned"]["phase_audit"].get(field)
        if b is None or c is None:
            continue
        ts = pd.Timestamp(row["trade_week"])
        records.append({"year": str(ts.year), "quarter": row["quarter"], "delta": float(c) - float(b)})
    frame = pd.DataFrame(records)
    if frame.empty:
        return {"year": {}, "quarter": {}}
    def agg(col: str) -> dict:
        out = {}
        for name, g in frame.groupby(col, sort=True):
            out[str(name)] = {"weeks": int(len(g)), "median_delta": float(g["delta"].median()), "mean_delta": float(g["delta"].mean()), "win_fraction": float((g["delta"] > 0).mean())}
        return out
    return {"year": agg("year"), "quarter": agg("quarter")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--risk-prereq-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prereq = verify_prerequisite(args.risk_prereq_receipt)
    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    op = opportunity_matrix(stitched)
    risk = build_risk_matrix(bars_1h(stitched))

    quarter_starts = list(pd.date_range(DISCOVERY_START, DISCOVERY_END, freq="QS", tz="UTC"))
    rows, fits = [], []
    for start, end in zip(quarter_starts[:-1], quarter_starts[1:]):
        fitted = fit_quarter(op, risk, start, end)
        if fitted is None:
            continue
        joined, receipt = fitted
        quarter = receipt["quarter"]
        rows.extend(weekly_rows(joined, quarter)); fits.append(receipt)
        print("QUARTER", quarter, json.dumps(receipt, sort_keys=True))

    by_week = {}
    for row in rows:
        key = row["trade_week"]
        if key not in by_week or row["rows"] > by_week[key]["rows"]:
            by_week[key] = row
    rows = [by_week[k] for k in sorted(by_week)]

    summaries = {str(c): paired_summary(rows, c) for c in POINT_COSTS}
    checks = {"1.0": advance_checks(summaries["1.0"]), "2.0": advance_checks(summaries["2.0"])}
    overall = checks["1.0"]["passed"] and checks["2.0"]["passed"]
    veto = {
        "long_signals": int(sum(f["long_signals"] for f in fits)), "short_signals": int(sum(f["short_signals"] for f in fits)),
        "long_vetoes": int(sum(f["long_vetoes"] for f in fits)), "short_vetoes": int(sum(f["short_vetoes"] for f in fits)),
    }
    veto["long_veto_rate"] = veto["long_vetoes"] / veto["long_signals"] if veto["long_signals"] else 0.0
    veto["short_veto_rate"] = veto["short_vetoes"] / veto["short_signals"] if veto["short_signals"] else 0.0

    result = {
        "schema": "foundry.mnq_ridge_h1_mae_orchestration_discovery.v1",
        "research_only": True, "promotion_authority": False,
        "contract": "research/mnq_ridge_h1_mae_orchestration_contract_20260901.json",
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}", "deep_timestamp_contract": contract_receipt(),
        "risk_prerequisite_commit": RISK_PREREQ_COMMIT, "risk_prerequisite": prereq,
        "discovery_period": {"start": DISCOVERY_START.isoformat(), "end_exclusive": DISCOVERY_END.isoformat()},
        "opportunity": "12Min H12 Ridge expected-move, prior-only inner OOF q50 magnitude gate",
        "risk_context": "1H H4 side-specific q80 MAE, prior-training prediction q80 veto threshold",
        "consumer": "veto only; never create/invert/resize a Ridge signal",
        "fit_receipts": fits, "weekly_rows": rows, "paired_summary": summaries,
        "advance_checks": checks,
        "decision": "advance_unchanged_to_corrected_2026_confirmation" if overall else "reject_or_redesign_consumer",
        "decomposition_after_1pt": decomposition(rows, 1.0), "decomposition_after_2pt": decomposition(rows, 2.0),
        "veto": veto,
        "nq_boundary": "NQ remains excluded until an unchanged composite survives corrected 2026 confirmation.",
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_RIDGE_H1_MAE_ORCHESTRATION_DISCOVERY=PASS")
    print("DECISION=" + result["decision"])
    print("CHECKS=" + json.dumps(checks, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
