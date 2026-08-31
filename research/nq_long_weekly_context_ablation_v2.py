from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_opportunity_target_matrix import classification, model, target_columns
from research.nq_cc0_multiyear_validation import load_normalized, bars12, phase_audit
from research.nq_long_weekly_context_ablation import LONG_WEEKLY_FEATURES, week_start_utc, load_long_weekly, long_context_table

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
TEST_END = pd.Timestamp("2025-10-06", tz="UTC")
ARMS = ("baseline20", "long_weekly_26y")


def summarize(rows: list[dict], arm: str) -> dict:
    vals = [r["arms"][arm]["nonoverlap_phase_audit"]["median_phase_net_after_2bp"] for r in rows]
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) < 80:
        raise RuntimeError(f"insufficient valid windows {arm}={len(vals)}")
    return {
        "windows": int(len(vals)),
        "positive_windows_net2": int(np.sum(vals > 0)),
        "positive_window_fraction_net2": float(np.mean(vals > 0)),
        "median_weekly_phase_median_net2": float(np.median(vals)),
        "mean_weekly_phase_median_net2": float(np.mean(vals)),
        "min_weekly_phase_median_net2": float(np.min(vals)),
        "max_weekly_phase_median_net2": float(np.max(vals)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--nq-normalized", type=Path, required=True)
    ap.add_argument("--nq-weekly", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    horizon, vol_mult = CONFIGS[args.config_key]

    intraday = load_normalized(args.nq_normalized)
    b = bars12(intraday)
    b["week_start"] = week_start_utc(b["timestamp"])
    b = b.merge(long_context_table(load_long_weekly(args.nq_weekly)), on="week_start", how="left", validate="many_to_one")
    if (b["long_context_source_week"].notna() & ~(b["long_context_source_week"] < b["week_start"])).any():
        raise RuntimeError("long weekly context leak")

    frame = _add_features(b)
    features = list(BASE_FEATURES) + LONG_WEEKLY_FEATURES
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    # 1999+ weekly history is fully warmed up before intraday source begins in 2022,
    # so this complete-case panel should not discard early NQ rows due to weekly warm-up.
    work = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    label, fwd, _ = target_columns(work, horizon, vol_mult)
    work["target"] = label
    work["fwd"] = fwd

    starts = list(pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"))
    rows = []
    for start in starts:
        end = start + pd.Timedelta(days=7)
        mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work["target"].notna()
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 300:
            continue
        test_start, test_stop = int(idx[0]), int(idx[-1] + 1)
        train_end = test_start - horizon
        if train_end < 2500:
            continue
        train = work.iloc[:train_end]
        train = train[train["target"].notna()]
        test = work.iloc[test_start:test_stop]
        test = test[(test["timestamp"] >= start) & (test["timestamp"] < end) & test["target"].notna()]
        if len(test) < 300 or train["timestamp"].max() >= test["timestamp"].min():
            continue
        y_test = test["target"].astype(int).to_numpy()
        fwd_test = test["fwd"].to_numpy(float)
        arms = {}
        for arm, cols in (("baseline20", list(BASE_FEATURES)), ("long_weekly_26y", features)):
            fitted = model().fit(train[cols].to_numpy(float), train["target"].astype(int).to_numpy())
            pred = fitted.predict(test[cols].to_numpy(float)).astype(int)
            arms[arm] = {
                "train_rows": int(len(train)),
                "classification": classification(y_test, pred),
                "predicted_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                "nonoverlap_phase_audit": phase_audit(test["timestamp"], pred, fwd_test, horizon),
            }
        rows.append({"start": start.isoformat(), "end": end.isoformat(), "test_rows": int(len(test)), "arms": arms})

    if len(rows) < 80:
        raise RuntimeError(f"insufficient comparable windows {len(rows)}")
    summary = {arm: summarize(rows, arm) for arm in ARMS}
    result = {
        "schema": "foundry.nq_long_weekly_context_ablation.v2",
        "research_only": True,
        "promotion_authority": False,
        "intraday_source": "tgtanalytics/nq-futures-1min-bar-2022-2025 CC0",
        "long_weekly_source": "youneseloiarm/nasdaq-cme-future-nq:NQ_in_weekly.csv CC0 1999+",
        "intraday_source_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "weekly_source_sha256": hashlib.sha256(args.nq_weekly.read_bytes()).hexdigest(),
        "config_key": args.config_key,
        "horizon": horizon,
        "vol_multiplier": vol_mult,
        "protocol": "baseline20 versus 1999+ NQ slow weekly state on one identical complete-case panel; weekly OHLCV withheld until following week; horizon purge before every OOS week; all non-overlapping UTC phase streams reported; no short-context warmup in row admission",
        "feature_sets": {"baseline20": list(BASE_FEATURES), "long_weekly_26y": features},
        "weekly_windows": rows,
        "summary": summary,
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_LONG_WEEKLY_CONTEXT_ABLATION=PASS")
    print(json.dumps(summary, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
