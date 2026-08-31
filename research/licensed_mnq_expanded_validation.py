from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import (
    BASE_FEATURES,
    EXPANDED_FEATURES,
    REGIME_FEATURES,
    _add_features,
    _challengers,
    _evaluate,
    _target_specs,
)

MONTH_ORDER = {c: i for i, c in enumerate("FGHJKMNQUVXZ", start=1)}
OUTRIGHT = re.compile(r"^MNQ(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})$")
EXPECTED_COLUMNS = [
    "timestamp", "rtype", "publisher_id", "instrument_id", "open", "high", "low", "close", "volume", "symbol"
]


def _contract_key(symbol: str) -> tuple[int, int]:
    m = OUTRIGHT.fullmatch(symbol)
    if not m:
        raise ValueError(symbol)
    y = m.group("year")
    year = 2000 + int(y) if len(y) == 2 else 2020 + int(y)
    return year, MONTH_ORDER[m.group("month")]


def _load_aggregate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(f"unexpected licensed MNQ schema: {list(frame.columns)}")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame = frame[frame["symbol"].map(lambda s: bool(OUTRIGHT.fullmatch(s)))].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for c in ["open", "high", "low", "close", "volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="raise")
    frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("no licensed MNQ outright rows")
    return frame


def _roll_schedule(frame: pd.DataFrame, confirmation_sessions: int = 2) -> pd.DataFrame:
    work = frame.copy()
    work["session"] = work["timestamp"].dt.strftime("%Y%m%d")
    daily = work.groupby(["session", "symbol"], as_index=False)["volume"].sum()
    contracts = sorted(daily["symbol"].unique(), key=_contract_key)
    sessions = sorted(daily["session"].unique())
    volume = {(r.session, r.symbol): float(r.volume) for r in daily.itertuples()}
    active = 0
    streak = 0
    pending = False
    rows = []
    for session in sessions:
        reason = "hold"
        if pending and active + 1 < len(contracts):
            active += 1; streak = 0; pending = False; reason = "volume_crossover_confirmed_prior_session"
        while active + 1 < len(contracts) and volume.get((session, contracts[active]), 0.0) <= 0 and volume.get((session, contracts[active + 1]), 0.0) > 0:
            active += 1; streak = 0; reason = "current_contract_unavailable"
        current = contracts[active]
        current_volume = volume.get((session, current), 0.0)
        next_contract = contracts[active + 1] if active + 1 < len(contracts) else None
        next_volume = volume.get((session, next_contract), 0.0) if next_contract else 0.0
        if current_volume <= 0:
            later = [c for c in contracts[active + 1:] if volume.get((session, c), 0.0) > 0]
            if later:
                active = contracts.index(later[0]); current = contracts[active]
                current_volume = volume.get((session, current), 0.0)
                next_contract = contracts[active + 1] if active + 1 < len(contracts) else None
                next_volume = volume.get((session, next_contract), 0.0) if next_contract else 0.0
                streak = 0; reason = "later_contract_availability_fallback"
        if current_volume > 0 and next_volume > current_volume:
            streak += 1
            if streak >= confirmation_sessions:
                pending = True
        elif current_volume > 0:
            streak = 0
        rows.append({"session": session, "selected_contract": current, "roll_reason": reason})
    out = pd.DataFrame(rows)
    keys = [_contract_key(s) for s in out["selected_contract"]]
    if keys != sorted(keys):
        raise RuntimeError("licensed MNQ roll schedule moved backward")
    return out


def _stitch(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["session"] = work["timestamp"].dt.strftime("%Y%m%d")
    selected = schedule.set_index("session")["selected_contract"].to_dict()
    work["selected_contract"] = work["session"].map(selected)
    work = work[work["symbol"] == work["selected_contract"]].copy()
    work = work.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
    if work["timestamp"].duplicated().any() or not work["timestamp"].is_monotonic_increasing:
        raise RuntimeError("licensed MNQ stitched timestamps not unique/increasing")
    return work


def _bars(stitched: pd.DataFrame) -> pd.DataFrame:
    w = stitched.set_index("timestamp")
    bars = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), observed_minutes=("close", "count"),
        source_contract=("symbol", "first"), source_contract_last=("symbol", "last"),
    )
    bars = bars[bars["observed_minutes"] > 0].copy()
    mixed = bars["source_contract"] != bars["source_contract_last"]
    dropped = int(mixed.sum())
    bars = bars.loc[~mixed].drop(columns=["source_contract_last"]).reset_index()
    bars["market"] = "MNQ"
    bars.attrs["dropped_roll_boundary_bars"] = dropped
    return bars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    raw = _load_aggregate(args.aggregate)
    schedule = _roll_schedule(raw)
    stitched = _stitch(raw, schedule)
    bars = _bars(stitched)
    frame = _add_features(bars)

    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES))
    modes = {
        "baseline20": BASE_FEATURES,
        "expanded": expanded,
        "expanded_regime": list(dict.fromkeys(expanded + REGIME_FEATURES)),
    }
    candidates = []
    current = {}
    for mode, features in modes.items():
        for target, horizon in _target_specs():
            ev = _evaluate(frame, features, target, horizon)
            if ev is None:
                continue
            row = {"feature_mode": mode, "target": target, "horizon_rows": horizon, **ev}
            candidates.append(row)
            if target == "target_dir_h12":
                current[mode] = ev
    candidates.sort(key=lambda x: (x["selection_score"], x["discovery_ba_floor"]), reverse=True)
    if not candidates:
        raise RuntimeError("licensed MNQ produced no evaluable candidates")
    best = candidates[0]
    challenge = _challengers(frame, modes[best["feature_mode"]], best["target"], int(best["horizon_rows"]))
    result = {
        "schema": "foundry.licensed_mnq_expanded_validation.v1",
        "research_only": True,
        "promotion_authority": False,
        "source_dataset": "brandenmorris/mnq-1m-q4-2022-q4-2025-ts-ohlcv-sym",
        "source_version": 1,
        "source_license": "CC BY 4.0",
        "raw_outright_rows": int(len(raw)),
        "stitched_minutes": int(len(stitched)),
        "bars_12min": int(len(bars)),
        "dropped_roll_boundary_bars": int(bars.attrs.get("dropped_roll_boundary_bars", 0)),
        "first_timestamp": bars["timestamp"].iloc[0].isoformat(),
        "last_timestamp": bars["timestamp"].iloc[-1].isoformat(),
        "feature_modes": {k: len(v) for k, v in modes.items()},
        "excluded_forward_aligned_features": ["chikou_span"],
        "current_target_mode_comparison": current,
        "best_discovery_candidate": best,
        "best_holdout_challengers": challenge,
        "top10": candidates[:10],
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LICENSED_MNQ_EXPANDED_VALIDATION=PASS")
    print("BEST=" + best["feature_mode"] + "/" + best["target"])
    print("DISCOVERY_BA=" + str(best["discovery_ba_mean"]))
    print("HOLDOUT_BA=" + str(best["holdout"]["balanced_accuracy"]))
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
