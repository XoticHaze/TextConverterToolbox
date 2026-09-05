from __future__ import annotations

"""Parity-gated public Stage-A industry discovery runner.

This is a hosted-compute adapter for the frozen #142 generic opportunity semantics.
It first reproduces the accepted E&P / A&D / Homebuilder development matrix on the
original 2026-09-01 decision calendar. Target-family symbols are not loaded until that
parity gate passes. External holdouts are never loaded by this Stage-A runner.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

START = "2014-01-01"
END = "2026-09-03"
TRAIN = ("CAT", "JPM", "UNH", "XOM", "COST", "MSFT")
FEATURES = [
    "mom5", "mom20", "mom60", "mom100", "mom20_z252", "mom20_accel5",
    "vol20", "vol20_z252", "distance_high60", "rs_qqq20", "rs_qqq60",
    "qqq_mom20", "qqq_mom100",
]
DELAY = 1
HOLD = 20
COST_BPS = 25.0
TAIL = 0.30
FOLDS = 6
MIN_TRAIN = 756
PURGE = 22
EVAL_FIRST_FOLD = 2
FAMILY_REQUIRED = 5
_CACHE: dict[str, pd.DataFrame] = {}


def _epoch(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def _load(symbol: str) -> pd.DataFrame:
    if symbol in _CACHE:
        return _CACHE[symbol].copy()
    query = urlencode({
        "period1": _epoch(START),
        "period2": _epoch(END),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    })
    request = Request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}",
        headers={"User-Agent": "Mozilla/5.0 research-foundry/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 fixed HTTPS host
        payload = json.loads(response.read().decode("utf-8"))
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"{symbol}: no chart result")
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    close = adjusted or (indicators.get("quote") or [{}])[0].get("close")
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        "price": pd.to_numeric(pd.Series(close), errors="coerce"),
    }).dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    _CACHE[symbol] = frame
    return frame.copy()


def _bounded(symbol: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = _load(symbol)
    frame = frame.loc[frame.timestamp <= cutoff].reset_index(drop=True)
    if frame.empty or frame.iloc[-1].timestamp != cutoff:
        last = None if frame.empty else frame.iloc[-1].timestamp.isoformat()
        raise RuntimeError(f"{symbol}: exact cutoff unavailable expected={cutoff.isoformat()} last={last}")
    return frame


def _global_folds(n: int) -> list[tuple[int, int]]:
    edges = np.linspace(MIN_TRAIN, n - (HOLD + DELAY + 1), FOLDS + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(FOLDS)]


def _engineer(
    prices: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    modeled_symbols: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, price in prices.items():
        df = pd.DataFrame({"timestamp": calendar, "price": price.to_numpy(float)})
        df["ret1"] = df.price.pct_change()
        for n in (5, 20, 60, 100):
            df[f"mom{n}"] = df.price.pct_change(n)
        df["vol20"] = df.ret1.rolling(20, min_periods=20).std(ddof=0)
        prior_mom = df.mom20.shift(1)
        prior_vol = df.vol20.shift(1)
        mom_mean = prior_mom.rolling(252, min_periods=126).mean()
        mom_std = prior_mom.rolling(252, min_periods=126).std(ddof=0).replace(0, np.nan)
        vol_mean = prior_vol.rolling(252, min_periods=126).mean()
        vol_std = prior_vol.rolling(252, min_periods=126).std(ddof=0).replace(0, np.nan)
        df["mom20_z252"] = (df.mom20 - mom_mean) / mom_std
        df["vol20_z252"] = (df.vol20 - vol_mean) / vol_std
        df["mom20_accel5"] = df.mom20 - df.mom20.shift(5)
        df["distance_high60"] = df.price / df.price.rolling(60, min_periods=60).max() - 1.0
        out[symbol] = df

    qqq = out["QQQ"]
    for symbol in modeled_symbols:
        df = out[symbol]
        df["rs_qqq20"] = df.mom20 - qqq.mom20
        df["rs_qqq60"] = df.mom60 - qqq.mom60
        df["qqq_mom20"] = qqq.mom20
        df["qqq_mom100"] = qqq.mom100
        df["enter20_after25_bps"] = (
            df.price.shift(-(DELAY + HOLD)) / df.price.shift(-DELAY) - 1.0
        ) * 10000.0 - COST_BPS
    return out


def _state_frame(
    symbols: tuple[str, ...],
    data: dict[str, pd.DataFrame],
    folds: list[tuple[int, int]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        df = data[symbol].copy()
        df["symbol"] = symbol
        df["signal_i"] = np.arange(len(df), dtype=int)
        fold_col = np.zeros(len(df), dtype=int)
        for fold, (start, stop) in enumerate(folds, 1):
            fold_col[start:max(start, stop - (DELAY + HOLD))] = fold
        df["fold"] = fold_col
        rows.append(df[df.fold > 0][["symbol", "signal_i", "fold", *FEATURES, "enter20_after25_bps"]])
    frame = pd.concat(rows, ignore_index=True)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["enter20_after25_bps"])


def _fit(train: pd.DataFrame):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    model.fit(train[FEATURES].to_numpy(float), train["enter20_after25_bps"].to_numpy(float))
    return model


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _evaluate_matrix(families: dict, cutoff_text: str) -> dict:
    cutoff = pd.Timestamp(cutoff_text)
    development: list[str] = []
    symbol_family: dict[str, str] = {}
    for family, spec in families.items():
        symbols = tuple(spec["development_universe"])
        if len(symbols) != 8 or len(set(symbols)) != 8:
            raise RuntimeError(f"{family}: development universe must contain exactly eight unique symbols")
        for symbol in symbols:
            if symbol in symbol_family or symbol in TRAIN:
                raise RuntimeError(f"duplicate/training development symbol {symbol}")
            symbol_family[symbol] = family
            development.append(symbol)
    dev_tuple = tuple(development)

    # Freeze the common selection calendar across all families before any ETF is loaded.
    selection_symbols = (*TRAIN, *dev_tuple, "QQQ")
    raw = {symbol: _bounded(symbol, cutoff) for symbol in selection_symbols}
    common_sets = [set(frame.timestamp) for frame in raw.values()]
    calendar = pd.DatetimeIndex(sorted(set.intersection(*common_sets)))
    if len(calendar) < 1500:
        raise RuntimeError(f"insufficient common selection calendar={len(calendar)}")
    if calendar[-1] != cutoff:
        raise RuntimeError(f"common calendar cutoff drift expected={cutoff.isoformat()} got={calendar[-1].isoformat()}")
    prices = {symbol: raw[symbol].set_index("timestamp").price.reindex(calendar) for symbol in raw}
    if any(series.isna().any() for series in prices.values()):
        raise RuntimeError("missing price on frozen common selection calendar")

    # Benchmarks are observational only and load after the selection calendar is immutable.
    benchmarks: dict[str, pd.Series] = {}
    for etf in sorted({str(spec["primary_industry_etf"]) for spec in families.values()}):
        series = _bounded(etf, cutoff).set_index("timestamp").price.reindex(calendar)
        if series.isna().any():
            raise RuntimeError(f"{etf}: missing rows on already-frozen common calendar")
        benchmarks[etf] = series

    modeled = (*TRAIN, *dev_tuple)
    data = _engineer(prices, calendar, modeled)
    folds = _global_folds(len(calendar))
    train_states = _state_frame(TRAIN, data, folds)
    test_states = _state_frame(dev_tuple, data, folds)

    family_records: dict[str, list[dict]] = {family: [] for family in families}
    per_symbol: list[dict] = []
    for symbol in dev_tuple:
        family = symbol_family[symbol]
        benchmark = str(families[family]["primary_industry_etf"])
        fold_rows: list[dict] = []
        all_primary: list[dict] = []
        for fold in range(EVAL_FIRST_FOLD, FOLDS + 1):
            start, _ = folds[fold - 1]
            train = train_states[train_states.signal_i < start - PURGE]
            test = test_states[(test_states.symbol == symbol) & (test_states.fold == fold)]
            if len(train) < 1000:
                raise RuntimeError(f"{symbol} fold {fold}: insufficient training rows={len(train)}")
            model = _fit(train)
            prediction = model.predict(test[FEATURES].to_numpy(float))
            chosen = test.loc[prediction > 0].copy()
            prior = data[symbol].iloc[:start - PURGE].mom20.dropna()
            if len(prior) < 250:
                raise RuntimeError(f"{symbol} fold {fold}: insufficient threshold support={len(prior)}")
            threshold = float(prior.quantile(1.0 - TAIL))
            primary = chosen.loc[chosen.mom20 < threshold]

            fold_primary: list[dict] = []
            for row in primary.itertuples(index=False):
                signal_i = int(row.signal_i)
                exec_i = signal_i + DELAY
                exit_i = exec_i + HOLD
                stock_gross = float(prices[symbol].iloc[exit_i] / prices[symbol].iloc[exec_i] - 1.0) * 10000.0
                etf_gross = float(benchmarks[benchmark].iloc[exit_i] / benchmarks[benchmark].iloc[exec_i] - 1.0) * 10000.0
                record = {
                    "symbol": symbol,
                    "family": family,
                    "benchmark": benchmark,
                    "fold": fold,
                    "signal_date": calendar[signal_i].isoformat(),
                    "entry_date": calendar[exec_i].isoformat(),
                    "exit_date": calendar[exit_i].isoformat(),
                    "stock_net25_bps": stock_gross - COST_BPS,
                    "stock_after25_minus_sector_etf_gross_bps": stock_gross - COST_BPS - etf_gross,
                    "sector_substitution50_bps": stock_gross - etf_gross - 50.0,
                }
                fold_primary.append(record)
                all_primary.append(record)
                family_records[family].append(record)
            fold_rows.append({
                "fold": fold,
                "primary_states": len(fold_primary),
                "stock_net25_mean_bps": _mean([r["stock_net25_bps"] for r in fold_primary]),
                "stock_after25_minus_sector_etf_mean_bps": _mean([r["stock_after25_minus_sector_etf_gross_bps"] for r in fold_primary]),
                "sector_substitution50_mean_bps": _mean([r["sector_substitution50_bps"] for r in fold_primary]),
            })

        net_mean = _mean([r["stock_net25_bps"] for r in all_primary])
        excess_mean = _mean([r["stock_after25_minus_sector_etf_gross_bps"] for r in all_primary])
        sub50_mean = _mean([r["sector_substitution50_bps"] for r in all_primary])
        positive_net_folds = sum(1 for row in fold_rows if row["stock_net25_mean_bps"] is not None and row["stock_net25_mean_bps"] > 0)
        positive_excess_folds = sum(1 for row in fold_rows if row["stock_after25_minus_sector_etf_mean_bps"] is not None and row["stock_after25_minus_sector_etf_mean_bps"] > 0)
        passed = bool(
            len(all_primary) >= 20
            and net_mean is not None and net_mean > 0
            and excess_mean is not None and excess_mean > 0
            and positive_net_folds >= 3
            and positive_excess_folds >= 3
        )
        per_symbol.append({
            "symbol": symbol,
            "family": family,
            "benchmark": benchmark,
            "primary_states": len(all_primary),
            "stock_net25_mean_bps": net_mean,
            "stock_after25_minus_sector_etf_mean_bps": excess_mean,
            "sector_substitution50_mean_bps": sub50_mean,
            "positive_stock_net_folds": positive_net_folds,
            "positive_sector_excess_folds": positive_excess_folds,
            "development_transport_pass": passed,
            "folds": fold_rows,
        })

    family_summary: dict[str, dict] = {}
    for family, spec in families.items():
        rows = [row for row in per_symbol if row["family"] == family]
        passing = sum(int(row["development_transport_pass"]) for row in rows)
        records = family_records[family]
        family_summary[family] = {
            "benchmark": str(spec["primary_industry_etf"]),
            "development_symbols": list(spec["development_universe"]),
            "external_holdouts_loaded": False,
            "passing_symbols": passing,
            "required_passing_symbols": FAMILY_REQUIRED,
            "development_family_gate_pass": passing >= FAMILY_REQUIRED,
            "aggregate_primary_states": len(records),
            "aggregate_stock_net25_mean_bps": _mean([r["stock_net25_bps"] for r in records]),
            "aggregate_stock_after25_minus_sector_etf_mean_bps": _mean([r["stock_after25_minus_sector_etf_gross_bps"] for r in records]),
            "aggregate_sector_substitution50_mean_bps": _mean([r["sector_substitution50_bps"] for r in records]),
        }

    return {
        "selection_calendar": {
            "rows": len(calendar),
            "first": calendar[0].isoformat(),
            "last": calendar[-1].isoformat(),
            "common_cutoff": cutoff.isoformat(),
            "sector_etfs_used_in_selection_calendar": False,
        },
        "training_universe": list(TRAIN),
        "per_symbol": per_symbol,
        "family_summary": family_summary,
        "families_eligible_for_external_holdout": [family for family, row in family_summary.items() if row["development_family_gate_pass"]],
        "external_holdouts_loaded": False,
    }


def _assert_parity(actual: dict, expected: dict) -> dict:
    checks: dict[str, bool] = {}
    for family, frozen in expected.items():
        row = actual["family_summary"][family]
        checks[f"{family}.passing_symbols"] = row["passing_symbols"] == frozen["passing_symbols"]
        checks[f"{family}.aggregate_primary_states"] = row["aggregate_primary_states"] == frozen["aggregate_primary_states"]
        for key in (
            "aggregate_stock_net25_mean_bps",
            "aggregate_stock_after25_minus_sector_etf_mean_bps",
            "aggregate_sector_substitution50_mean_bps",
        ):
            checks[f"{family}.{key}"] = abs(float(row[key]) - float(frozen[key])) <= 1e-8
    if not all(checks.values()):
        failed = {key: value for key, value in checks.items() if not value}
        raise RuntimeError(f"STAGE_A_PARITY_FAILED {json.dumps(failed, sort_keys=True)}")
    return checks


def run(contract_path: Path, output_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    generic = contract["generic_prior"]
    if tuple(generic["training_universe"]) != TRAIN or list(generic["features"]) != FEATURES:
        raise RuntimeError("generic prior identity drift")
    if int(generic["delay_sessions"]) != DELAY or int(generic["hold_sessions"]) != HOLD:
        raise RuntimeError("execution horizon drift")
    if float(generic["cost_bps"]) != COST_BPS or float(generic["tail"]) != TAIL:
        raise RuntimeError("cost/tail drift")
    if int(generic["minimum_train_index"]) != MIN_TRAIN or int(generic["purge_sessions"]) != PURGE:
        raise RuntimeError("chronology drift")
    if generic["ticker_identity"] is not False or generic["test_symbol_target_rows_used_in_training"] is not False:
        raise RuntimeError("target-exclusion drift")
    boundaries = contract["boundaries"]
    for key in ("threshold_search", "hyperparameter_search", "external_holdouts_loaded", "runtime_mutation", "broker_action", "promotion_authority", "live_trading_change"):
        if boundaries[key] is not False:
            raise RuntimeError(f"unsafe/drifted boundary {key}")

    _CACHE.clear()
    parity = _evaluate_matrix(contract["parity_control"]["families"], contract["parity_control"]["common_cutoff"])
    parity_checks = _assert_parity(parity, contract["parity_control"]["accepted_family_summary"])

    # New target families are not loaded before accepted #142 parity succeeds.
    _CACHE.clear()
    target = _evaluate_matrix(contract["target"]["families"], contract["target"]["common_cutoff"])
    result = {
        "schema": "foundry.research.public_runner_industry_stagea_result.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_authority": contract["scientific_authority"],
        "parity": {"passed": True, "checks": parity_checks, "result": parity},
        "target": {"name": contract["target"]["name"], "result": target},
        "role_separation": contract["role_separation"],
        "boundaries": {**boundaries, "parity_required_before_target_load": True, "parity_passed": True, "target_external_holdouts_loaded": False},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.contract), Path(args.output))


if __name__ == "__main__":
    main()
