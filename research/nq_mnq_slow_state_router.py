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
from research.nq_mnq_slow_state_stratification import load_daily, slow_state_table, trade_week_key, purged_train
from research.nq_to_mnq_execution_transfer import phase_audit

CONFIGS = {"h12_vol10": (12, 1.0), "h24_vol05": (24, 0.5)}
TEST_START = pd.Timestamp("2023-07-02", tz="UTC")
DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")
TEST_END = pd.Timestamp("2025-12-08", tz="UTC")
STATE_AXES = ("trend_state", "vol_state", "drawdown_state")
MODEL_ARMS = ("nq_expanding", "mnq_expanding", "equal_market_pooled")
STATIC_ARMS = ("always_long", "always_short")
CANDIDATE_ARMS = MODEL_ARMS + STATIC_ARMS
COST_FIELDS = {
    "0p5": "median_phase_net_points_after_0p5pt",
    "1p0": "median_phase_net_points_after_1p0pt",
    "2p0": "median_phase_net_points_after_2p0pt",
    "4p0": "median_phase_net_points_after_4p0pt",
}
SELECTION_FIELD = COST_FIELDS["1p0"]
MIN_DISCOVERY_STATE_WEEKS = 8
MIN_CONFIRMATION_WEEKS = 20


def build_weekly_rows(config_key: str, deep_root: Path, nq_normalized: Path, nq_daily: Path) -> list[dict]:
    horizon, vol_mult = CONFIGS[config_key]
    features = list(BASE_FEATURES)
    mnq_raw = load_deep(deep_root)
    mnq = _add_features(deep_bars(stitch_deep(mnq_raw, deep_roll_schedule(mnq_raw))))
    nq = _add_features(bars12(load_normalized(nq_normalized)))
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    mnq = mnq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nq = nq[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    mnq_target, _, _ = target_columns(mnq, horizon, vol_mult)
    nq_target, _, _ = target_columns(nq, horizon, vol_mult)
    mnq["target"] = mnq_target
    nq["target"] = nq_target
    mnq["point_move"] = mnq["close"].shift(-horizon) - mnq["close"]

    state = slow_state_table(load_daily(nq_daily))
    state_map = {pd.Timestamp(r.trade_week): r for r in state.itertuples(index=False)}
    starts = list(pd.date_range(TEST_START, TEST_END - pd.Timedelta(days=7), freq="7D", tz="UTC"))
    rows: list[dict] = []
    for start in starts:
        end = start + pd.Timedelta(days=7)
        local_key = trade_week_key(pd.Series([start]))[0]
        state_row = state_map.get(pd.Timestamp(local_key))
        if state_row is None or state_row.trend_state == "unknown" or state_row.vol_state == "unknown" or state_row.drawdown_state == "unknown":
            continue
        test = mnq[(mnq["timestamp"] >= start) & (mnq["timestamp"] < end) & mnq["target"].notna() & mnq["point_move"].notna()].copy()
        if len(test) < 300:
            continue
        mnq_train = purged_train(mnq, start, horizon)
        nq_train = purged_train(nq, start, horizon)
        if len(mnq_train) < 50000 or len(nq_train) < 10000:
            continue
        if mnq_train["timestamp"].max() >= test["timestamp"].min() or nq_train["timestamp"].max() >= test["timestamp"].min():
            raise RuntimeError("chronology overlap")
        fitted = {
            "nq_expanding": model().fit(nq_train[features].to_numpy(float), nq_train["target"].astype(int).to_numpy()),
            "mnq_expanding": model().fit(mnq_train[features].to_numpy(float), mnq_train["target"].astype(int).to_numpy()),
            "equal_market_pooled": fit_equal_market(features, mnq_train, nq_train),
        }
        y = test["target"].astype(int).to_numpy()
        move = test["point_move"].to_numpy(float)
        arms = {}
        for arm, fitted_model in fitted.items():
            pred = fitted_model.predict(test[features].to_numpy(float)).astype(int)
            arms[arm] = {
                "classification": classification(y, pred),
                "phase_audit": phase_audit(test["timestamp"], pred, move, horizon),
            }
        for arm, pred in {"always_long": np.ones(len(test), dtype=int), "always_short": -np.ones(len(test), dtype=int)}.items():
            arms[arm] = {"classification": classification(y, pred), "phase_audit": phase_audit(test["timestamp"], pred, move, horizon)}
        rows.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "trend_state": state_row.trend_state,
            "vol_state": state_row.vol_state,
            "drawdown_state": state_row.drawdown_state,
            "state_source_week": pd.Timestamp(state_row.source_week).isoformat(),
            "arms": arms,
        })
    if len(rows) < 80:
        raise RuntimeError(f"insufficient weekly rows {len(rows)}")
    return rows


def learn_mapping(discovery: list[dict], axis: str) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    states = sorted({r[axis] for r in discovery})
    for state in states:
        subset = [r for r in discovery if r[axis] == state]
        candidates = []
        for arm in CANDIDATE_ARMS:
            vals = [r["arms"][arm]["phase_audit"].get(SELECTION_FIELD) for r in subset]
            arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
            if len(arr) < MIN_DISCOVERY_STATE_WEEKS:
                continue
            median = float(np.median(arr))
            mean = float(np.mean(arr))
            pos = float(np.mean(arr > 0))
            # A state arm must be positive by both central tendency measures and
            # profitable in a majority of its discovery weeks. Otherwise abstain.
            if median > 0 and mean > 0 and pos >= 0.55:
                candidates.append({"arm": arm, "weeks": int(len(arr)), "median": median, "mean": mean, "positive_fraction": pos})
        if candidates:
            candidates.sort(key=lambda x: (x["median"], x["positive_fraction"], x["mean"], x["arm"]), reverse=True)
            chosen = candidates[0]
            mapping[state] = {"selected_arm": chosen["arm"], "eligible": candidates}
        else:
            mapping[state] = {"selected_arm": "abstain", "eligible": []}
    return mapping


def summarize_confirmation(rows: list[dict], axis: str, mapping: dict[str, dict]) -> dict:
    routed: dict[str, list[float]] = {k: [] for k in COST_FIELDS}
    static: dict[str, dict[str, list[float]]] = {arm: {k: [] for k in COST_FIELDS} for arm in CANDIDATE_ARMS}
    decisions = []
    for r in rows:
        state = r[axis]
        selected = mapping.get(state, {"selected_arm": "abstain"})["selected_arm"]
        decisions.append({"start": r["start"], "state": state, "selected_arm": selected})
        if selected != "abstain":
            for ck, field in COST_FIELDS.items():
                v = r["arms"][selected]["phase_audit"].get(field)
                if v is not None and np.isfinite(v):
                    routed[ck].append(float(v))
        for arm in CANDIDATE_ARMS:
            for ck, field in COST_FIELDS.items():
                v = r["arms"][arm]["phase_audit"].get(field)
                if v is not None and np.isfinite(v):
                    static[arm][ck].append(float(v))

    def stats(vals: list[float], total_weeks: int) -> dict:
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            return {"traded_weeks": 0, "total_weeks": total_weeks}
        k = max(1, int(np.ceil(0.10 * len(arr))))
        worst = np.sort(arr)[:k]
        return {
            "traded_weeks": int(len(arr)),
            "total_weeks": int(total_weeks),
            "coverage": float(len(arr) / total_weeks),
            "positive_weeks": int(np.sum(arr > 0)),
            "positive_fraction": float(np.mean(arr > 0)),
            "median_points": float(np.median(arr)),
            "mean_points": float(np.mean(arr)),
            "p10_points": float(np.quantile(arr, 0.10)),
            "bottom10pct_mean_points": float(np.mean(worst)),
            "worst_week_points": float(np.min(arr)),
            "best_week_points": float(np.max(arr)),
        }

    result = {
        "total_confirmation_weeks": int(len(rows)),
        "decisions": decisions,
        "routed": {ck: stats(vals, len(rows)) for ck, vals in routed.items()},
        "static_comparators": {arm: {ck: stats(vals, len(rows)) for ck, vals in by_cost.items()} for arm, by_cost in static.items()},
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--nq-normalized", type=Path, required=True)
    ap.add_argument("--nq-daily", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = build_weekly_rows(args.config_key, args.deep_root, args.nq_normalized, args.nq_daily)
    discovery = [r for r in rows if pd.Timestamp(r["start"]) < DISCOVERY_END]
    confirmation = [r for r in rows if pd.Timestamp(r["start"]) >= DISCOVERY_END]
    if len(discovery) < 50 or len(confirmation) < MIN_CONFIRMATION_WEEKS:
        raise RuntimeError(f"insufficient nested split discovery={len(discovery)} confirmation={len(confirmation)}")

    routers = {}
    for axis in STATE_AXES:
        mapping = learn_mapping(discovery, axis)
        routers[axis] = {
            "mapping": mapping,
            "confirmation": summarize_confirmation(confirmation, axis, mapping),
        }

    result = {
        "schema": "foundry.nq_mnq_slow_state_router.v1",
        "research_only": True,
        "promotion_authority": False,
        "execution_target": "MNQ",
        "config_key": args.config_key,
        "horizon": CONFIGS[args.config_key][0],
        "vol_multiplier": CONFIGS[args.config_key][1],
        "feature_set": "baseline20; slow state routes only after discovery and never enters prediction",
        "state_axes": list(STATE_AXES),
        "candidate_arms": list(CANDIDATE_ARMS) + ["abstain"],
        "selection_contract": {
            "discovery_period": f"{TEST_START.isoformat()} -> {DISCOVERY_END.isoformat()}",
            "confirmation_period": f"{DISCOVERY_END.isoformat()} -> {TEST_END.isoformat()}",
            "per_state_min_discovery_weeks": MIN_DISCOVERY_STATE_WEEKS,
            "eligibility": "at 1 MNQ point cost, discovery median > 0, discovery mean > 0, and positive-week fraction >= 0.55; otherwise abstain",
            "choice": "among eligible arms choose highest discovery median, then positive fraction, then mean; freeze mapping before confirmation",
            "no_confirmation_tuning": True,
        },
        "discovery_weeks": int(len(discovery)),
        "confirmation_weeks": int(len(confirmation)),
        "routers": routers,
        "nq_normalized_sha256": hashlib.sha256(args.nq_normalized.read_bytes()).hexdigest(),
        "nq_daily_sha256": hashlib.sha256(args.nq_daily.read_bytes()).hexdigest(),
        "excluded_forward_aligned_features": ["chikou_span"],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("NQ_MNQ_SLOW_STATE_ROUTER=PASS")
    print(json.dumps({axis: routers[axis]["confirmation"]["routed"]["1p0"] for axis in STATE_AXES}, sort_keys=True))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
