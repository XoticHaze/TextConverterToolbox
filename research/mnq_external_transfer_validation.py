from __future__ import annotations

import re

import pandas as pd

CONTRACT_RE = re.compile(r"^MNQ (?P<month>\d{2})-(?P<year>\d{2})$")


def contract_key(name: str) -> tuple[int, int]:
    match = CONTRACT_RE.fullmatch(name)
    if not match:
        raise ValueError(name)
    return 2000 + int(match.group("year")), int(match.group("month"))


def deep_roll_schedule(frame: pd.DataFrame, confirmation_sessions: int = 2) -> pd.DataFrame:
    work = frame.copy()
    work["session"] = work["timestamp"].dt.strftime("%Y%m%d")
    daily = work.groupby(["session", "symbol"], as_index=False)["volume"].sum()
    contracts = sorted(daily["symbol"].unique(), key=contract_key)
    sessions = sorted(daily["session"].unique())
    volume = {(row.session, row.symbol): float(row.volume) for row in daily.itertuples()}
    active = 0
    streak = 0
    pending = False
    rows = []
    for session in sessions:
        reason = "hold"
        if pending and active + 1 < len(contracts):
            active += 1
            streak = 0
            pending = False
            reason = "volume_crossover_confirmed_prior_session"
        while active + 1 < len(contracts) and volume.get((session, contracts[active]), 0.0) <= 0 and volume.get((session, contracts[active + 1]), 0.0) > 0:
            active += 1
            streak = 0
            reason = "current_contract_unavailable"
        current = contracts[active]
        current_volume = volume.get((session, current), 0.0)
        next_contract = contracts[active + 1] if active + 1 < len(contracts) else None
        next_volume = volume.get((session, next_contract), 0.0) if next_contract else 0.0
        if current_volume <= 0:
            later = [c for c in contracts[active + 1:] if volume.get((session, c), 0.0) > 0]
            if later:
                active = contracts.index(later[0])
                current = contracts[active]
                current_volume = volume.get((session, current), 0.0)
                next_contract = contracts[active + 1] if active + 1 < len(contracts) else None
                next_volume = volume.get((session, next_contract), 0.0) if next_contract else 0.0
                streak = 0
                reason = "later_contract_availability_fallback"
        if current_volume > 0 and next_volume > current_volume:
            streak += 1
            if streak >= confirmation_sessions:
                pending = True
        elif current_volume > 0:
            streak = 0
        rows.append({"session": session, "selected_contract": current, "roll_reason": reason})
    out = pd.DataFrame(rows)
    keys = [contract_key(symbol) for symbol in out["selected_contract"]]
    if keys != sorted(keys):
        raise RuntimeError("deep MNQ roll schedule moved backward")
    return out


def stitch_deep(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["session"] = work["timestamp"].dt.strftime("%Y%m%d")
    selected = schedule.set_index("session")["selected_contract"].to_dict()
    work["selected_contract"] = work["session"].map(selected)
    work = work[work["symbol"] == work["selected_contract"]].copy()
    work = work.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
    if work["timestamp"].duplicated().any() or not work["timestamp"].is_monotonic_increasing:
        raise RuntimeError("deep MNQ stitched timestamps not unique/increasing")
    return work
