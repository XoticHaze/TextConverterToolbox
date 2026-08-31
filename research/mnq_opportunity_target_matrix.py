from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars

HORIZONS = (6, 12, 24)
VOL_MULTIPLIERS = (0.5, 1.0)
COST_FLOOR = 0.0002
MAX_HORIZON = max(HORIZONS)


def model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def target_columns(frame: pd.DataFrame, horizon: int, mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = frame["close"].astype(float)
    fwd_simple = close.shift(-horizon) / close - 1.0
    fwd_log = np.log(close.shift(-horizon) / close)
    threshold = np.maximum(COST_FLOOR, mult * frame["rv_120"].astype(float) * math.sqrt(horizon))
    label = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = fwd_log.notna() & pd.Series(threshold, index=frame.index).notna()
    label.loc[valid] = 0.0
    label.loc[valid & (fwd_log > threshold)] = 1.0
    label.loc[valid & (fwd_log < -threshold)] = -1.0
    return label, fwd_simple, pd.Series(threshold, index=frame.index)


def quarter_starts() -> list[pd.Timestamp]:
    return list(pd.date_range("2022-01-01", "2026-04-01", freq="QS", tz="UTC"))


def classification(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def economic(pred: np.ndarray, fwd: np.ndarray) -> dict | None:
    selected = pred != 0
    if selected.sum() < 100:
        return None
    signed = np.where(pred[selected] == 1, 1.0, -1.0) * fwd[selected]
    return {
        "signals": int(selected.sum()),
        "coverage": float(selected.mean()),
        "long_signals": int((pred[selected] == 1).sum()),
        "short_signals": int((pred[selected] == -1).sum()),
        "gross_mean_forward_return": float(np.mean(signed)),
        "gross_median_forward_return": float(np.median(signed)),
        "gross_positive_rate": float(np.mean(signed > 0)),
        "net_mean_after_2bp": float(np.mean(signed - 0.0002)),
        "net_mean_after_5bp": float(np.mean(signed - 0.0005)),
        "net_mean_after_10bp": float(np.mean(signed - 0.0010)),
        "note": "overlapping research signals; cost sensitivity only, not executable strategy PnL",
    }


def summarize_quarters(rows: list[dict]) -> dict:
    ba = [r["classification"]["balanced_accuracy"] for r in rows]
    net2 = [r["economic"]["net_mean_after_2bp"] for r in rows if r["economic"]]
    gross = [r["economic"]["gross_mean_forward_return"] for r in rows if r["economic"]]
    return {
        "quarters": len(rows),
        "mean_balanced_accuracy": float(np.mean(ba)),
        "median_balanced_accuracy": float(np.median(ba)),
        "min_balanced_accuracy": float(np.min(ba)),
        "max_balanced_accuracy": float(np.max(ba)),
        "quarters_net_positive_after_2bp": int(sum(v > 0 for v in net2)),
        "median_quarter_net_after_2bp": float(np.median(net2)) if net2 else None,
        "median_quarter_gross": float(np.median(gross)) if gross else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = load_deep(args.deep_root)
    stitched = stitch_deep(raw, deep_roll_schedule(raw))
    bars = deep_bars(stitched)
    frame = _add_features(bars)

    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    feature_sets = {
        "baseline20": list(BASE_FEATURES),
        "expanded_regime": expanded,
    }
    common_cols = list(dict.fromkeys(["timestamp", "close", "rv_120", *expanded]))
    work = frame[common_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    configs = []
    for h in HORIZONS:
        for mult in VOL_MULTIPLIERS:
            key = f"h{h}_vol{str(mult).replace('.', '')}"
            label, fwd, threshold = target_columns(work, h, mult)
            work[f"target_{key}"] = label
            work[f"fwd_{key}"] = fwd
            work[f"threshold_{key}"] = threshold
            configs.append({"key": key, "horizon": h, "vol_multiplier": mult})

    quarters = quarter_starts()
    result_configs: dict[str, dict] = {}

    for cfg in configs:
        key = cfg["key"]
        target_col = f"target_{key}"
        fwd_col = f"fwd_{key}"
        threshold_col = f"threshold_{key}"
        cfg_result = {
            **cfg,
            "threshold": "max(2bp, vol_multiplier * causal rv_120 * sqrt(horizon))",
            "feature_sets": {},
        }

        for feature_name, features in feature_sets.items():
            rows = []
            all_truth: list[int] = []
            all_pred: list[int] = []
            all_fwd: list[float] = []
            all_threshold: list[float] = []

            for i in range(len(quarters) - 1):
                start, end = quarters[i], quarters[i + 1]
                test_mask = (work["timestamp"] >= start) & (work["timestamp"] < end) & work[target_col].notna()
                test_idx = np.flatnonzero(test_mask.to_numpy())
                if len(test_idx) < 500:
                    continue
                test_start = int(test_idx[0])
                test_end = int(test_idx[-1] + 1)
                train_end = test_start - cfg["horizon"]
                if train_end < 10000:
                    continue

                train = work.iloc[:train_end]
                train = train[train[target_col].notna()]
                test = work.iloc[test_start:test_end]
                test = test[test[target_col].notna()]
                if len(train) < 10000 or len(test) < 500:
                    continue

                y_train = train[target_col].astype(int).to_numpy()
                y_test = test[target_col].astype(int).to_numpy()
                if len(np.unique(y_train)) < 3 or len(np.unique(y_test)) < 3:
                    continue

                m = model().fit(train[features].to_numpy(float), y_train)
                pred = m.predict(test[features].to_numpy(float)).astype(int)
                fwd = test[fwd_col].to_numpy(float)
                econ = economic(pred, fwd)
                row = {
                    "period": f"{start.year}Q{((start.month - 1)//3)+1}",
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_last_timestamp": train["timestamp"].iloc[-1].isoformat(),
                    "test_first_timestamp": test["timestamp"].iloc[0].isoformat(),
                    "test_last_timestamp": test["timestamp"].iloc[-1].isoformat(),
                    "target_class_counts": {str(c): int((y_test == c).sum()) for c in (-1, 0, 1)},
                    "predicted_class_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "classification": classification(y_test, pred),
                    "economic": econ,
                    "median_causal_threshold": float(np.median(test[threshold_col].to_numpy(float))),
                }
                rows.append(row)
                all_truth.extend(y_test.tolist())
                all_pred.extend(pred.tolist())
                all_fwd.extend(fwd.tolist())
                all_threshold.extend(test[threshold_col].to_numpy(float).tolist())
                print(key, feature_name, row["period"], "BA", row["classification"]["balanced_accuracy"], "NET2", None if econ is None else econ["net_mean_after_2bp"], "COV", None if econ is None else econ["coverage"])

            if len(rows) < 8:
                raise RuntimeError(f"{key}/{feature_name}: insufficient outer quarters {len(rows)}")
            truth = np.asarray(all_truth, dtype=int)
            pred = np.asarray(all_pred, dtype=int)
            fwd = np.asarray(all_fwd, dtype=float)
            cfg_result["feature_sets"][feature_name] = {
                "quarter_rows": rows,
                "stability": summarize_quarters(rows),
                "aggregate": {
                    "classification": classification(truth, pred),
                    "economic": economic(pred, fwd),
                    "target_class_counts": {str(c): int((truth == c).sum()) for c in (-1, 0, 1)},
                    "predicted_class_counts": {str(c): int((pred == c).sum()) for c in (-1, 0, 1)},
                    "median_causal_threshold": float(np.median(np.asarray(all_threshold, dtype=float))),
                },
            }

        result_configs[key] = cfg_result

    result = {
        "schema": "foundry.mnq_opportunity_target_matrix.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176",
        "protocol": "predeclared 3x2 target matrix; quarterly expanding past-only refit; horizon purge; fixed outer timestamps; no confidence helper or post-hoc threshold tuning",
        "target_semantics": "three-class down/no-trade/up outcome using causal realized-volatility-scaled threshold with 2bp floor",
        "excluded_forward_aligned_features": ["chikou_span"],
        "deep_source_rows": int(len(raw)),
        "stitched_minutes": int(len(stitched)),
        "bars_12min": int(len(bars)),
        "configs": result_configs,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MNQ_OPPORTUNITY_TARGET_MATRIX=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
