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
from research.nq_to_mnq_execution_transfer import phase_audit

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-12-08", tz="UTC")
ARMS = ("nq_expanding", "mnq_expanding", "equal_market_pooled")
BASELINES = ("always_long", "always_short")
POINT_COSTS = (0.5, 1.0, 2.0, 4.0)


def summarize(rows: list[dict], arm: str) -> dict:
    summaries = {}
    for cost in POINT_COSTS:
        key = str(cost).replace(".", "p")
        field = f"median_phase_net_points_after_{key}pt"
        vals = [r["arms"][arm]["phase_audit"].get(field) for r in rows]
        vals = np.asarray([v for v in vals if v is not None], dtype=float)
        if len(vals) < 80:
            raise RuntimeError(f"insufficient {arm}/{key} windows {len(vals)}")
        summaries[f"after_{key}pt"] = {
            "windows": int(len(vals)),
            "positive_windows": int(np.sum(vals > 0)),
            "positive_window_fraction": float(np.mean(vals > 0)),
            "median_weekly_phase_median_points": float(np.median(vals)),
            "mean_weekly_phase_median_points": float(np.mean(vals)),
            "min_weekly_phase_median_points": float(np.min(vals)),
            "max_weekly_phase_median_points": float(np.max(vals)),
            "median_weekly_phase_median_mnq_dollars": float(np.median(vals) * 2.0),
        }
    return summaries


def purged_train(frame: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    before = frame[(frame["timestamp"] < start) & frame["target"].notna()].copy()
    if len(before) <= horizon:
        return before.iloc[0:0]
    return before.iloc[:-horizon].copy()


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
    nq = _add_features(bars12(load_normalized(args.nq_normalized)))
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    mnq_target, _, _ = target_columns(mnq, horizon, vol_mult)
    nq_target, _, _ = target_columns(nq, horizon, vol_mult)
    mnq["target"] = mnq_target
    nq["target"] = nq_target
    mnq["point_move"] = mnq["close"].shift(-horizon) - mnq["close"]

    starts = list(pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"))
    rows = []
    for start in starts:
        end = start + pd.Timedelta(days=7)
        test = mnq[(mnq["timestamp"] >= start) & (mnq["timestamp"] < end) & mnq["target"].notna() & mnq["point_move"].notna()].copy()
        if len(test) < 300:
            continue
        mnq_train = purged_train(mnq, start, horizon)
        nq_train = purged_train(nq, start, horizon)
        if len(mnq_train) < 50000 or len(nq_train) < 10000:
            continue
        if mnq_train["timestamp"].max() >= test["timestamp"].min() or nq_train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError(f"chronology overlap {start.date()}")
        if len(np.unique(mnq_train["target"].astype(int))) < 3 or len(np.unique(nq_train["target"].astype(int))) < 3:
            continue

        y_test = test["target"].astype(int).to_numpy()
        point_move = test["point_move"].to_numpy(float)
        fitted = {
            "nq_expanding": model().fit(nq_train[features].to_numpy(float), nq_train["target"].astype(int).to_numpy()),
            "mnq_expanding": model().fit(mnq_train[features].to_numpy(float), mnq_train["target"].astype(int).to_numpy()),
            "equal_market_pooled": fit_equal_market(features, mnq_train, nq_train),
        }
        arms = {}
        for arm, fitted_model in fitted.items():
            pred = fitted_model.predict(test[features].to_numpy(float)).astype(int)
            arms[arm] = {
                "classification": classification(y_test, pred),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "phase_audit": phase_audit(test["timestamp"], pred, point_move, horizon),
            }
        for arm, pred in {
            "always_long": np.ones(len(test), dtype=int),
            "always_short": -np.ones(len(test), dtype=int),
        }.items():
            arms[arm] = {
                "classification": classification(y_test, pred),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "phase_audit": phase_audit(test["timestamp"], pred, point_move, horizon),
            }
        rows.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "test_rows": int(len(test)),
            "mnq_train_rows": int(len(mnq_train)),
            "nq_train_rows": int(len(nq_train)),
            "mnq_train_last_timestamp": mnq_train["timestamp"].max().isoformat(),
            "nq_train_last_timestamp": nq_train["timestamp"].max().isoformat(),
            "arms": arms,
        })
        print(args.config_key, start.date(), {a: arms[a]["phase_audit"].get("median_phase_net_points_after_1p0pt") for a in (*ARMS, *BASELINES)})

    if len(rows) < 80:
        raise RuntimeError(f"insufficient MNQ weekly OOS windows {len(rows)}")
    summary = {arm: summarize(rows, arm) for arm in (*ARMS, *BASELINES)}
    result = {
        "schema": "foundry.nq_mnq_weekly_execution_target.v1",
        "research_only": True,
        "promotion_authority": False,
        "execution_target": "MNQ",
        "mnq_multiplier_dollars_per_point": 2.0,
        "mnq_source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "nq_source": "tgtanalytics/nq-futures-1min-bar-2022-2025 CC0",
        "nq_normalized_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "feature_set": "baseline20",
        "point_cost_grid": list(POINT_COSTS),
        "protocol": "weekly MNQ OOS from 2023-07 through 2025-12; before every future week independently refit NQ-only expanding, MNQ-only expanding, and equal-total-market-weight pooled models with horizon purge; evaluate all arms on identical MNQ timestamps using all non-overlapping UTC phase streams and MNQ point/$ economics; always-long/always-short baselines reported on same timestamps",
        "weekly_windows": rows,
        "summary": summary,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_MNQ_WEEKLY_EXECUTION_TARGET=PASS")
    print(json.dumps(summary, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
