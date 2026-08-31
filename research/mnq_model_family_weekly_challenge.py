from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_opportunity_target_matrix import classification, target_columns
from research.nq_to_mnq_execution_transfer import phase_audit

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
FAMILIES = ("logistic", "hist_gradient_boosting", "extra_trees")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)


def make_model(name: str):
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ])
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=42,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=160,
            max_features="sqrt",
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    raise RuntimeError(f"unknown model family {name}")


def fit_model(name: str, x: np.ndarray, y: np.ndarray):
    fitted = make_model(name)
    if name == "hist_gradient_boosting":
        fitted.fit(x, y, sample_weight=compute_sample_weight(class_weight="balanced", y=y))
    else:
        fitted.fit(x, y)
    return fitted


def trade_week_key(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="raise")
    local = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    trade_date = local.dt.normalize() + pd.to_timedelta((local.dt.hour >= 18).astype(int), unit="D")
    return trade_date - pd.to_timedelta(trade_date.dt.weekday, unit="D")


def summary(rows: list[dict], family: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        vals = np.asarray([r["families"][family]["phase_audit"].get(field) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 80:
            raise RuntimeError(f"insufficient weekly economics for {family}/{key}: {len(vals)}")
        out[f"after_{key}pt"] = {
            "weeks": int(len(vals)),
            "positive_weeks": int(np.sum(vals > 0)),
            "positive_week_fraction": float(np.mean(vals > 0)),
            "median_weekly_phase_median_points": float(np.median(vals)),
            "mean_weekly_phase_median_points": float(np.mean(vals)),
            "median_weekly_phase_median_mnq_dollars": float(2.0 * np.median(vals)),
            "min_weekly_phase_median_points": float(np.min(vals)),
            "max_weekly_phase_median_points": float(np.max(vals)),
        }
    return out


def paired_summary(rows: list[dict], challenger: str, reference: str = "logistic") -> dict:
    result = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        diffs = []
        for row in rows:
            a = row["families"][challenger]["phase_audit"].get(field)
            b = row["families"][reference]["phase_audit"].get(field)
            if a is not None and b is not None:
                diffs.append(float(a) - float(b))
        arr = np.asarray(diffs, dtype=float)
        result[f"after_{key}pt"] = {
            "weeks": int(len(arr)),
            "median_challenger_minus_logistic_points": float(np.median(arr)),
            "mean_challenger_minus_logistic_points": float(np.mean(arr)),
            "challenger_win_fraction": float(np.mean(arr > 0)),
            "tie_fraction": float(np.mean(arr == 0)),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]
    features = list(BASE_FEATURES)

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    work = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, _, _ = target_columns(work, horizon, vol_mult)
    work["target"] = label
    work["point_move"] = work["close"].shift(-horizon) - work["close"]
    work["trade_week"] = trade_week_key(work["timestamp"])

    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    weekly_rows: list[dict] = []
    fit_receipts: list[dict] = []

    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        test_mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna() & work["point_move"].notna()
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < 2000:
            continue
        test_start = int(test_idx[0])
        train_end = test_start - horizon
        if train_end < 50000:
            continue
        train = work.iloc[:train_end]
        train = train[train["target"].notna()]
        test = work.iloc[int(test_idx[0]):int(test_idx[-1] + 1)]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["target"].notna() & test["point_move"].notna()].copy()
        if len(train) < 50000 or len(test) < 2000:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("quarter chronology overlap")
        y_train = train["target"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 3:
            continue

        fitted = {name: fit_model(name, train[features].to_numpy(float), y_train) for name in FAMILIES}
        preds = {name: m.predict(test[features].to_numpy(float)).astype(int) for name, m in fitted.items()}
        fit_receipts.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
        })

        for week_key, positions in test.groupby("trade_week", sort=True).groups.items():
            pos = np.asarray(list(positions), dtype=int)
            # groupby indices are original work indices; select through work to keep exact alignment.
            week = work.loc[pos]
            week = week[(week["timestamp"] >= start) & (week["timestamp"] < end) & week["target"].notna() & week["point_move"].notna()]
            if len(week) < 300:
                continue
            point_move = week["point_move"].to_numpy(float)
            y_week = week["target"].astype(int).to_numpy()
            families = {}
            for name in FAMILIES:
                pred = fitted[name].predict(week[features].to_numpy(float)).astype(int)
                families[name] = {
                    "classification": classification(y_week, pred),
                    "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "phase_audit": phase_audit(week["timestamp"], pred, point_move, horizon),
                }
            weekly_rows.append({
                "trade_week": pd.Timestamp(week_key).isoformat(),
                "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
                "rows": int(len(week)),
                "families": families,
            })

    # A partial first/last trade week can occur at quarter boundaries; keep one record per exact trade week.
    by_week: dict[str, dict] = {}
    for row in weekly_rows:
        key = row["trade_week"]
        if key in by_week:
            # Prefer the record with more rows; exact equal-size duplicates are a contract error.
            if row["rows"] == by_week[key]["rows"]:
                raise RuntimeError(f"ambiguous duplicate trade week {key}")
            if row["rows"] > by_week[key]["rows"]:
                by_week[key] = row
        else:
            by_week[key] = row
    rows = [by_week[k] for k in sorted(by_week)]
    if len(rows) < 100:
        raise RuntimeError(f"insufficient unique weekly OOS rows: {len(rows)}")

    result = {
        "schema": "foundry.mnq_model_family_weekly_challenge.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "feature_set": "baseline20",
        "families": list(FAMILIES),
        "model_contract": {
            "logistic": "StandardScaler + class-balanced LogisticRegression(max_iter=2000)",
            "hist_gradient_boosting": "HistGradientBoostingClassifier learning_rate=.05, max_iter=180, max_leaf_nodes=31, min_samples_leaf=40, balanced sample weights",
            "extra_trees": "ExtraTreesClassifier 160 trees, sqrt features, min_samples_leaf=20, class_weight=balanced",
        },
        "protocol": "fixed quarterly past-only refits from 2023-07 through 2025-12 with horizon purge; identical MNQ future rows for all model families; report every futures trade week using all non-overlapping UTC phase streams; point economics at 0.5/1/2/4 MNQ points; no hyperparameter tuning or family selection on OOS",
        "fit_receipts": fit_receipts,
        "weekly_rows": rows,
        "summary": {name: summary(rows, name) for name in FAMILIES},
        "paired_vs_logistic": {name: paired_summary(rows, name) for name in FAMILIES if name != "logistic"},
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_MODEL_FAMILY_WEEKLY_CHALLENGE=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print(json.dumps(result["paired_vs_logistic"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
