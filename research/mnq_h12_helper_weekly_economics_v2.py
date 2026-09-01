from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import research.mnq_h12_helper_weekly_economics as v1
from research.mnq_model_family_weekly_challenge import make_model as family_model

HORIZON = 12
VOL_MULTIPLIER = 1.0
TEST_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
MIN_COMPLETE_WEEKS = 80
TRUST_QUANTILE = 0.65
SOURCE_COMMIT = "fc5508e2c152938d6d9eb70a36b888ae26107176"
POLICIES = v1.POLICIES
STATE_FEATURES = v1.STATE_FEATURES

# These are not newly chosen targets. They are immutable identity anchors from the
# already-completed 83-week H12 family panel at TextConverterToolbox@5fafa571....
BASELINE_ANCHOR = {
    "weeks": 83,
    "after_1p0pt": {
        "median": 1.8127705627705628,
        "mean": -0.0782087773836909,
        "positive_week_fraction": 0.5542168674698795,
    },
    "after_2p0pt": {
        "median": 0.8127705627705628,
        "mean": -1.078208777383691,
        "positive_week_fraction": 0.5180722891566265,
    },
}


def base_model() -> Pipeline:
    # Reuse the exact baseline family contract rather than a helper-specific clone.
    return family_model("logistic")


def gate_model() -> Pipeline:
    # Median values are learned from prior helper-training rows only. This preserves
    # the baseline20 row universe instead of deleting observations for helper-only
    # feature missingness.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                C=0.3,
                random_state=43,
            ),
        ),
    ])


def base_confidence(model: Pipeline, x: np.ndarray) -> np.ndarray:
    return np.max(model.predict_proba(x), axis=1)


def inner_oof_meta(train: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    """Generate four strictly-past base-model OOF blocks inside the outer train set."""
    n = len(train)
    first_test = max(20000, n // 3)
    remaining = n - first_test
    fold_size = remaining // 4
    if fold_size < 3000:
        raise RuntimeError(f"insufficient inner OOF span: n={n} fold_size={fold_size}")

    parts: list[pd.DataFrame] = []
    for fold in range(4):
        test_start = first_test + fold * fold_size
        test_end = n if fold == 3 else first_test + (fold + 1) * fold_size
        fit_end = test_start - HORIZON
        if fit_end < 15000 or test_end - test_start < 3000:
            raise RuntimeError(
                f"invalid inner OOF fold {fold}: fit_end={fit_end} test_rows={test_end-test_start}"
            )
        fit = train.iloc[:fit_end]
        hold = train.iloc[test_start:test_end]
        y_fit = fit["target"].astype(int).to_numpy()
        if len(np.unique(y_fit)) < 3:
            raise RuntimeError(f"inner OOF fold {fold} lacks three classes")

        model = base_model().fit(fit[base_features].to_numpy(float), y_fit)
        x_hold = hold[base_features].to_numpy(float)
        pred = model.predict(x_hold).astype(int)
        conf = base_confidence(model, x_hold)

        part = hold[["timestamp", *STATE_FEATURES]].copy()
        part["base_pred"] = pred
        part["base_confidence"] = conf
        part["correct"] = (pred == hold["target"].astype(int).to_numpy()).astype(int)
        part["inner_fold"] = fold
        parts.append(part)

    meta = pd.concat(parts, ignore_index=True)
    if len(meta) < 12000 or meta["correct"].nunique() < 2:
        raise RuntimeError(f"invalid helper meta corpus rows={len(meta)} classes={meta['correct'].nunique()}")
    return meta


def gate_oof_cut(meta: pd.DataFrame, gate_features: list[str]) -> tuple[float, list[dict]]:
    """Fit the trust cut from helper probabilities that are themselves chronological OOF."""
    probabilities: list[np.ndarray] = []
    receipts: list[dict] = []
    for fold in (1, 2, 3):
        fit = meta[meta["inner_fold"] < fold]
        hold = meta[meta["inner_fold"] == fold]
        if len(fit) < 3000 or len(hold) < 1000 or fit["correct"].nunique() < 2:
            raise RuntimeError(f"invalid gate OOF fold {fold}: fit={len(fit)} hold={len(hold)}")
        model = gate_model().fit(fit[gate_features].to_numpy(float), fit["correct"].to_numpy(int))
        prob = model.predict_proba(hold[gate_features].to_numpy(float))[:, 1]
        probabilities.append(prob)
        receipts.append({
            "fold": fold,
            "fit_rows": int(len(fit)),
            "hold_rows": int(len(hold)),
            "fit_last_timestamp": pd.to_datetime(fit["timestamp"], utc=True).max().isoformat(),
            "hold_first_timestamp": pd.to_datetime(hold["timestamp"], utc=True).min().isoformat(),
        })
    p = np.concatenate(probabilities)
    return float(np.quantile(p, TRUST_QUANTILE)), receipts


def economic_summary(values: list[float | None]) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"weeks": 0}
    k = max(1, int(np.ceil(0.10 * len(arr))))
    worst = np.sort(arr)[:k]
    cumulative = np.cumsum(arr)
    curve = np.concatenate(([0.0], cumulative))
    peak = np.maximum.accumulate(curve)
    drawdown = curve - peak
    positive_total = float(np.sum(arr[arr > 0]))
    top5 = np.sort(arr[arr > 0])[-5:] if np.any(arr > 0) else np.asarray([], dtype=float)
    return {
        "weeks": int(len(arr)),
        "positive_weeks": int(np.sum(arr > 0)),
        "positive_week_fraction": float(np.mean(arr > 0)),
        "median_weekly_phase_median_points": float(np.median(arr)),
        "mean_weekly_phase_median_points": float(np.mean(arr)),
        "p10_weekly_phase_median_points": float(np.quantile(arr, 0.10)),
        "bottom10pct_mean_points": float(np.mean(worst)),
        "worst_week_points": float(np.min(arr)),
        "best_week_points": float(np.max(arr)),
        "cumulative_weekly_phase_median_points": float(np.sum(arr)),
        "max_drawdown_weekly_phase_median_points": float(np.min(drawdown)),
        "top5_positive_share": float(np.sum(top5) / positive_total) if positive_total > 0 else None,
    }


def summarize(rows: list[dict], policy: str) -> dict:
    out: dict = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        out[f"after_{key}pt"] = economic_summary(
            [row["policies"][policy]["phase_audit"].get(field) for row in rows]
        )
    coverage = np.asarray([row["policies"][policy]["coverage"] for row in rows], dtype=float)
    out["coverage"] = {
        "median": float(np.median(coverage)),
        "mean": float(np.mean(coverage)),
        "min": float(np.min(coverage)),
        "max": float(np.max(coverage)),
    }
    return out


def paired(rows: list[dict], challenger: str) -> dict:
    out: dict = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        diffs: list[float] = []
        for row in rows:
            a = row["policies"][challenger]["phase_audit"].get(field)
            b = row["policies"]["baseline_logistic"]["phase_audit"].get(field)
            if a is not None and b is not None:
                diffs.append(float(a) - float(b))
        arr = np.asarray(diffs, dtype=float)
        if len(arr) == 0:
            out[f"after_{key}pt"] = {"weeks": 0}
            continue
        k = max(1, int(np.ceil(0.10 * len(arr))))
        out[f"after_{key}pt"] = {
            "weeks": int(len(arr)),
            "median_delta_points": float(np.median(arr)),
            "mean_delta_points": float(np.mean(arr)),
            "p10_delta_points": float(np.quantile(arr, 0.10)),
            "bottom10pct_mean_delta_points": float(np.mean(np.sort(arr)[:k])),
            "win_fraction": float(np.mean(arr > 0)),
            "loss_fraction": float(np.mean(arr < 0)),
            "tie_fraction": float(np.mean(arr == 0)),
        }
    return out


def assert_baseline_identity(summary: dict) -> None:
    base = summary["baseline_logistic"]
    for cost_key, anchor in (("after_1p0pt", BASELINE_ANCHOR["after_1p0pt"]), ("after_2p0pt", BASELINE_ANCHOR["after_2p0pt"])):
        row = base[cost_key]
        if row.get("weeks") != BASELINE_ANCHOR["weeks"]:
            raise RuntimeError(f"baseline weekly identity mismatch: {cost_key} weeks={row.get('weeks')}")
        checks = {
            "median": row["median_weekly_phase_median_points"],
            "mean": row["mean_weekly_phase_median_points"],
            "positive_week_fraction": row["positive_week_fraction"],
        }
        for name, value in checks.items():
            if not np.isclose(value, anchor[name], atol=1e-10, rtol=0):
                raise RuntimeError(
                    f"baseline identity mismatch {cost_key}/{name}: got={value} expected={anchor[name]}"
                )


def decision(summary: dict, paired_result: dict, challenger: str) -> dict:
    failures: list[dict] = []
    for cost_key in ("after_1p0pt", "after_2p0pt"):
        base = summary["baseline_logistic"][cost_key]
        helper = summary[challenger][cost_key]
        delta = paired_result[challenger][cost_key]
        checks = {
            "helper_support_80_weeks": helper.get("weeks", 0) >= MIN_COMPLETE_WEEKS,
            "paired_support_80_weeks": delta.get("weeks", 0) >= MIN_COMPLETE_WEEKS,
        }
        if checks["helper_support_80_weeks"] and checks["paired_support_80_weeks"]:
            checks.update({
                "paired_median_delta_positive": delta["median_delta_points"] > 0,
                "paired_win_fraction_above_half": delta["win_fraction"] > 0.50,
                "positive_week_fraction_not_worse": helper["positive_week_fraction"] >= base["positive_week_fraction"],
                "p10_not_worse": helper["p10_weekly_phase_median_points"] >= base["p10_weekly_phase_median_points"],
                "bottom10pct_mean_not_worse": helper["bottom10pct_mean_points"] >= base["bottom10pct_mean_points"],
                "max_drawdown_not_worse": helper["max_drawdown_weekly_phase_median_points"] >= base["max_drawdown_weekly_phase_median_points"],
            })
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append({"cost": cost_key, "failed": failed})
    return {
        "action": "advance_to_nq_external_validation" if not failures else "reject_or_rework_helper",
        "failures": failures,
        "rule": "all frozen support, paired-economic, and tail checks must pass at both 1pt and 2pt",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    base_features = list(v1.BASE_FEATURES)
    all_columns = list(dict.fromkeys(["timestamp", "close", "rv_120", *base_features, *STATE_FEATURES]))

    raw = v1.load_deep(args.deep_root)
    stitched = v1.stitch_deep(raw, v1.deep_roll_schedule(raw))
    bars = v1.deep_bars(stitched)
    frame = v1._add_features(bars)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # IMPORTANT: select rows using ONLY the pre-existing baseline20 contract.
    # Helper-only state missingness is handled inside gate_model and cannot alter
    # the baseline/OOS week universe.
    baseline_required = ["timestamp", "close", "rv_120", *base_features]
    baseline_mask = frame[baseline_required].notna().all(axis=1)
    work = frame.loc[baseline_mask, all_columns].copy().reset_index(drop=True)

    label, _, _ = v1.target_columns(work, HORIZON, VOL_MULTIPLIER)
    work["target"] = label
    work["point_move"] = work["close"].shift(-HORIZON) - work["close"]
    work["trade_week"] = v1.trade_week_key(work["timestamp"])

    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    weekly_rows: list[dict] = []
    fit_receipts: list[dict] = []

    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        test_mask = (
            (work["timestamp"] >= start)
            & (work["timestamp"] < end)
            & work["target"].notna()
            & work["point_move"].notna()
        )
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < 2000:
            continue
        test_start_idx = int(test_idx[0])
        train_end = test_start_idx - HORIZON
        if train_end < 50000:
            continue
        train = work.iloc[:train_end]
        train = train[train["target"].notna()].copy()
        test = work.iloc[int(test_idx[0]) : int(test_idx[-1] + 1)]
        test = test[
            (test["timestamp"] >= start)
            & (test["timestamp"] < end)
            & test["target"].notna()
            & test["point_move"].notna()
        ].copy()
        if len(train) < 50000 or len(test) < 2000:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("outer chronology overlap")

        y_train = train["target"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 3:
            continue

        meta = inner_oof_meta(train, base_features)
        gate_features = [*STATE_FEATURES, "base_confidence", "base_pred"]
        gate_cut, gate_oof_receipts = gate_oof_cut(meta, gate_features)
        confidence_cut = float(np.quantile(meta["base_confidence"].to_numpy(float), TRUST_QUANTILE))
        gate = gate_model().fit(meta[gate_features].to_numpy(float), meta["correct"].to_numpy(int))
        base = base_model().fit(train[base_features].to_numpy(float), y_train)

        fit_receipts.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "meta_rows": int(len(meta)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
            "test_first_timestamp": test["timestamp"].min().isoformat(),
            "test_last_timestamp": test["timestamp"].max().isoformat(),
            "confidence_q65": confidence_cut,
            "local_state_gate_q65_from_gate_oof": gate_cut,
            "gate_oof_receipts": gate_oof_receipts,
        })

        for week_key, positions in test.groupby("trade_week", sort=True).groups.items():
            pos = np.asarray(list(positions), dtype=int)
            week = work.loc[pos]
            week = week[
                (week["timestamp"] >= start)
                & (week["timestamp"] < end)
                & week["target"].notna()
                & week["point_move"].notna()
            ].copy()
            if len(week) < 300:
                continue

            x_week = week[base_features].to_numpy(float)
            baseline_pred = base.predict(x_week).astype(int)
            baseline_conf = base_confidence(base, x_week)
            gate_frame = week[STATE_FEATURES].copy()
            gate_frame["base_confidence"] = baseline_conf
            gate_frame["base_pred"] = baseline_pred
            gate_prob = gate.predict_proba(gate_frame[gate_features].to_numpy(float))[:, 1]

            predictions = {
                "baseline_logistic": baseline_pred,
                "confidence_trust_q65": np.where(baseline_conf >= confidence_cut, baseline_pred, 0).astype(int),
                "local_state_trust_q65": np.where(gate_prob >= gate_cut, baseline_pred, 0).astype(int),
            }
            y_week = week["target"].astype(int).to_numpy()
            point_move = week["point_move"].to_numpy(float)
            policies = {}
            for name, pred in predictions.items():
                policies[name] = {
                    "classification": v1.classification(y_week, pred),
                    "coverage": float(np.mean(pred != 0)),
                    "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "phase_audit": v1.phase_audit(week["timestamp"], pred, point_move, HORIZON),
                }
            weekly_rows.append({
                "trade_week": pd.Timestamp(week_key).isoformat(),
                "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
                "rows": int(len(week)),
                "policies": policies,
            })

    by_week: dict[str, dict] = {}
    for row in weekly_rows:
        key = row["trade_week"]
        if key in by_week:
            if row["rows"] == by_week[key]["rows"]:
                raise RuntimeError(f"ambiguous duplicate trade week {key}")
            if row["rows"] > by_week[key]["rows"]:
                by_week[key] = row
        else:
            by_week[key] = row
    rows = [by_week[key] for key in sorted(by_week)]

    summaries = {name: summarize(rows, name) for name in POLICIES}
    assert_baseline_identity(summaries)
    paired_results = {name: paired(rows, name) for name in POLICIES if name != "baseline_logistic"}
    decisions = {name: decision(summaries, paired_results, name) for name in POLICIES if name != "baseline_logistic"}

    result = {
        "schema": "foundry.mnq_h12_helper_weekly_economics.v2",
        "research_only": True,
        "promotion_authority": False,
        "source": f"mbytes21/MNQ_DATA@{SOURCE_COMMIT}",
        "baseline_source": "TextConverterToolbox@5fafa571590cfde33a7f4167b3b2972690cc3a59:h12_vol10",
        "baseline_identity": "PASS",
        "timeframe": "12Min",
        "horizon": HORIZON,
        "vol_multiplier": VOL_MULTIPLIER,
        "minimum_complete_weeks": MIN_COMPLETE_WEEKS,
        "policies": list(POLICIES),
        "helper_action": "abstain_only",
        "helper_missingness": "training-only median imputation inside gate pipeline; never changes baseline row universe",
        "helper_threshold": "q65 from chronology-clean prior helper/base OOF probabilities; never selected from outer OOS economics",
        "protocol": "quarterly past-only outer walk-forward with H12 purge; exact pre-existing baseline20 OOS universe; four inner chronological base OOF blocks; helper trust cut from nested chronological gate OOF; all non-overlapping UTC H12 phases; no direction inversion; no OOS tuning; no contemporaneous NQ features",
        "fit_receipts": fit_receipts,
        "weekly_rows": rows,
        "summary": summaries,
        "paired_vs_baseline": paired_results,
        "predeclared_decision": decisions,
        "nq_validation_boundary": "NQ remains external/pre-2019 validation only after an MNQ helper passes the frozen MNQ rule",
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("MNQ_H12_HELPER_WEEKLY_ECONOMICS_V2=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print(json.dumps(result["paired_vs_baseline"], sort_keys=True))
    print(json.dumps(result["predeclared_decision"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
