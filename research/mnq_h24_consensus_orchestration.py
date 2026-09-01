from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_model_family_weekly_challenge import fit_model, trade_week_key
from research.mnq_opportunity_target_matrix import classification, target_columns
from research.nq_to_mnq_execution_transfer import phase_audit

HORIZON = 24
VOL_MULT = 0.5
TEST_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
MIN_COMPLETE_WEEKS = 80
# Selective consensus/veto policies can legitimately have too few signals for a
# phase-level weekly economic statistic in some otherwise-valid OOS weeks. The
# experiment still requires >=80 unique OOS weeks overall; this separate floor
# only governs whether a selective policy has enough valid weekly observations
# to report distribution/tail statistics. It does not change predictions or
# select an OOS policy.
MIN_VALID_POLICY_WEEKS = 60
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)
POLICIES = (
    "logistic",
    "extra_trees",
    "consensus_only",
    "logistic_opposite_veto",
)


def build_policy(name: str, logistic: np.ndarray, extra: np.ndarray) -> np.ndarray:
    if name == "logistic":
        return logistic.copy()
    if name == "extra_trees":
        return extra.copy()
    if name == "consensus_only":
        return np.where((logistic == extra) & (logistic != 0), logistic, 0).astype(int)
    if name == "logistic_opposite_veto":
        opposite = (logistic != 0) & (extra != 0) & (logistic == -extra)
        out = logistic.copy()
        out[opposite] = 0
        return out.astype(int)
    raise RuntimeError(name)


def tail_summary(vals: np.ndarray, total_oos_weeks: int) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < MIN_VALID_POLICY_WEEKS:
        raise RuntimeError(f"insufficient valid policy tail rows {len(vals)}")
    k = max(1, int(np.ceil(0.10 * len(vals))))
    worst = np.sort(vals)[:k]
    return {
        "weeks": int(len(vals)),
        "total_oos_weeks": int(total_oos_weeks),
        "valid_week_fraction": float(len(vals) / total_oos_weeks),
        "positive_weeks": int(np.sum(vals > 0)),
        "positive_week_fraction": float(np.mean(vals > 0)),
        "median_weekly_points": float(np.median(vals)),
        "mean_weekly_points": float(np.mean(vals)),
        "p10_weekly_points": float(np.quantile(vals, 0.10)),
        "bottom10pct_mean_points": float(np.mean(worst)),
        "worst_week_points": float(np.min(vals)),
        "best_week_points": float(np.max(vals)),
    }


def summarize(rows: list[dict], policy: str) -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        vals = [r["policies"][policy]["phase_audit"].get(field) for r in rows]
        out[f"after_{key}pt"] = tail_summary(np.asarray(vals, dtype=float), len(rows))
    cov = np.asarray([r["policies"][policy]["coverage"] for r in rows], dtype=float)
    out["coverage"] = {
        "median": float(np.median(cov)),
        "mean": float(np.mean(cov)),
        "min": float(np.min(cov)),
        "max": float(np.max(cov)),
    }
    return out


def paired(rows: list[dict], challenger: str, reference: str = "logistic") -> dict:
    out = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        diffs = []
        for r in rows:
            a = r["policies"][challenger]["phase_audit"].get(field)
            b = r["policies"][reference]["phase_audit"].get(field)
            if a is not None and b is not None:
                diffs.append(float(a) - float(b))
        arr = np.asarray(diffs, dtype=float)
        if len(arr) < MIN_VALID_POLICY_WEEKS:
            raise RuntimeError(f"insufficient paired valid rows {challenger}/{key}: {len(arr)}")
        out[f"after_{key}pt"] = {
            "weeks": int(len(arr)),
            "total_oos_weeks": int(len(rows)),
            "valid_week_fraction": float(len(arr) / len(rows)),
            "median_delta_points": float(np.median(arr)),
            "mean_delta_points": float(np.mean(arr)),
            "win_fraction": float(np.mean(arr > 0)),
            "loss_fraction": float(np.mean(arr < 0)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    features = list(BASE_FEATURES)
    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    work = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, _, _ = target_columns(work, HORIZON, VOL_MULT)
    work["target"] = label
    work["point_move"] = work["close"].shift(-HORIZON) - work["close"]
    work["trade_week"] = trade_week_key(work["timestamp"])

    quarter_starts = list(pd.date_range(TEST_START, TEST_END, freq="QS", tz="UTC"))
    weekly_rows = []
    fits = []
    for qi in range(len(quarter_starts) - 1):
        start, end = quarter_starts[qi], quarter_starts[qi + 1]
        mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna() & work["point_move"].notna()
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 2000:
            continue
        test_start = int(idx[0])
        train_end = test_start - HORIZON
        if train_end < 50000:
            continue
        train = work.iloc[:train_end]
        train = train[train["target"].notna()]
        test = work.iloc[int(idx[0]):int(idx[-1] + 1)]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["target"].notna() & test["point_move"].notna()].copy()
        if len(train) < 50000 or len(test) < 2000:
            continue
        if train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("chronology overlap")
        y_train = train["target"].astype(int).to_numpy()
        logistic_model = fit_model("logistic", train[features].to_numpy(float), y_train)
        extra_model = fit_model("extra_trees", train[features].to_numpy(float), y_train)
        fits.append({
            "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
            "train_rows": int(len(train)),
            "train_last_timestamp": train["timestamp"].max().isoformat(),
            "test_rows": int(len(test)),
        })

        for week_key, positions in test.groupby("trade_week", sort=True).groups.items():
            pos = np.asarray(list(positions), dtype=int)
            week = work.loc[pos]
            week = week[(week["timestamp"] >= start) & (week["timestamp"] < end) & week["target"].notna() & week["point_move"].notna()]
            if len(week) < 300:
                continue
            x = week[features].to_numpy(float)
            log_pred = logistic_model.predict(x).astype(int)
            ext_pred = extra_model.predict(x).astype(int)
            y = week["target"].astype(int).to_numpy()
            move = week["point_move"].to_numpy(float)
            policies = {}
            for policy in POLICIES:
                pred = build_policy(policy, log_pred, ext_pred)
                policies[policy] = {
                    "coverage": float(np.mean(pred != 0)),
                    "classification": classification(y, pred),
                    "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "phase_audit": phase_audit(week["timestamp"], pred, move, HORIZON),
                }
            weekly_rows.append({
                "trade_week": pd.Timestamp(week_key).isoformat(),
                "quarter": f"{start.year}Q{((start.month - 1)//3)+1}",
                "rows": int(len(week)),
                "policies": policies,
            })

    by_week = {}
    for r in weekly_rows:
        key = r["trade_week"]
        if key in by_week:
            if r["rows"] == by_week[key]["rows"]:
                raise RuntimeError(f"ambiguous duplicate week {key}")
            if r["rows"] > by_week[key]["rows"]:
                by_week[key] = r
        else:
            by_week[key] = r
    rows = [by_week[k] for k in sorted(by_week)]
    if len(rows) < MIN_COMPLETE_WEEKS:
        raise RuntimeError(f"insufficient unique OOS weeks {len(rows)}")

    result = {
        "schema": "foundry.mnq_h24_consensus_orchestration.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "feature_set": "baseline20",
        "horizon": HORIZON,
        "vol_multiplier": VOL_MULT,
        "models": {
            "logistic": "same fixed class-balanced logistic contract from model-family challenge",
            "extra_trees": "same fixed ExtraTrees contract from model-family challenge",
        },
        "policies": {
            "logistic": "anchor model unchanged",
            "extra_trees": "challenger model unchanged",
            "consensus_only": "trade only when both models predict the same nonzero direction; otherwise abstain",
            "logistic_opposite_veto": "use logistic prediction except abstain when ExtraTrees predicts the opposite nonzero direction; ExtraTrees neutral does not veto",
        },
        "reporting_contract": {
            "minimum_total_oos_weeks": MIN_COMPLETE_WEEKS,
            "minimum_valid_selective_policy_weeks": MIN_VALID_POLICY_WEEKS,
            "reason": "selective policies can lack enough phase-level signals in some otherwise valid OOS weeks; missing weekly economics are reported through valid_week_fraction and never imputed",
        },
        "protocol": "policies predeclared from completed model-family evidence; quarterly past-only refits with H24 purge; identical MNQ OOS rows; all trade weeks and all non-overlapping UTC phase streams reported; no OOS threshold fitting, policy selection, or state gate",
        "fit_receipts": fits,
        "weekly_rows": rows,
        "summary": {p: summarize(rows, p) for p in POLICIES},
        "paired_vs_logistic": {p: paired(rows, p) for p in POLICIES if p != "logistic"},
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_H24_CONSENSUS_ORCHESTRATION=PASS")
    print(json.dumps(result["summary"], sort_keys=True))
    print(json.dumps(result["paired_vs_logistic"], sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
