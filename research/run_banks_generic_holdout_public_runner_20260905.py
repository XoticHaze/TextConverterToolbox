from __future__ import annotations

"""Approved public-runner consumer for the frozen generic industry holdout logic.

This module reproduces the accepted #146 holdout evaluator semantics from its frozen
source lineage.  It MUST first reproduce the accepted Homebuilder holdout decision and
rounded aggregate economics.  Only after that parity gate passes may the prospectively
frozen Bank holdouts be loaded/evaluated.
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


def _evaluate_family(holdout: tuple[str, ...], benchmark: str) -> dict:
    selection_symbols = (*TRAIN, *holdout, "QQQ")
    raw = {symbol: _load(symbol) for symbol in selection_symbols}
    cutoff = min(df.iloc[-1].timestamp for df in raw.values())
    common_sets = [set(df.loc[df.timestamp <= cutoff, "timestamp"]) for df in raw.values()]
    calendar = pd.DatetimeIndex(sorted(set.intersection(*common_sets)))
    if len(calendar) < 1500:
        raise RuntimeError(f"insufficient holdout calendar={len(calendar)}")
    prices = {
        symbol: raw[symbol].set_index("timestamp").price.reindex(calendar)
        for symbol in raw
    }
    if any(series.isna().any() for series in prices.values()):
        raise RuntimeError("missing price on frozen holdout selection calendar")

    # The benchmark is observational only. It is loaded after the selection calendar is frozen.
    benchmark_price = _load(benchmark).set_index("timestamp").price.reindex(calendar)
    if benchmark_price.isna().any():
        raise RuntimeError(f"{benchmark}: missing rows on already-frozen holdout calendar")

    modeled = (*TRAIN, *holdout)
    data = _engineer(prices, calendar, modeled)
    folds = _global_folds(len(calendar))
    train_states = _state_frame(TRAIN, data, folds)
    holdout_states = _state_frame(holdout, data, folds)

    per_symbol: list[dict] = []
    all_records: list[dict] = []
    for symbol in holdout:
        fold_rows: list[dict] = []
        symbol_records: list[dict] = []
        for fold in range(EVAL_FIRST_FOLD, FOLDS + 1):
            start, _ = folds[fold - 1]
            train = train_states[train_states.signal_i < start - PURGE]
            test = holdout_states[(holdout_states.symbol == symbol) & (holdout_states.fold == fold)]
            if len(train) < 1000:
                raise RuntimeError(f"{symbol} fold {fold}: insufficient training rows={len(train)}")
            model = _fit(train)
            prediction = model.predict(test[FEATURES].to_numpy(float))
            chosen = test.loc[prediction > 0].copy()

            prior = data[symbol].iloc[:start - PURGE].mom20.dropna()
            if len(prior) < 250:
                raise RuntimeError(f"{symbol} fold {fold}: insufficient threshold history")
            threshold = float(prior.quantile(1.0 - TAIL))
            primary = chosen.loc[chosen.mom20 < threshold]

            fold_records: list[dict] = []
            for row in primary.itertuples(index=False):
                signal_i = int(row.signal_i)
                exec_i = signal_i + DELAY
                exit_i = exec_i + HOLD
                stock_gross = float(prices[symbol].iloc[exit_i] / prices[symbol].iloc[exec_i] - 1.0) * 10000.0
                benchmark_gross = float(benchmark_price.iloc[exit_i] / benchmark_price.iloc[exec_i] - 1.0) * 10000.0
                record = {
                    "symbol": symbol,
                    "fold": fold,
                    "signal_date": calendar[signal_i].isoformat(),
                    "entry_date": calendar[exec_i].isoformat(),
                    "exit_date": calendar[exit_i].isoformat(),
                    "stock_net25_bps": stock_gross - 25.0,
                    "stock_after25_minus_benchmark_gross_bps": stock_gross - 25.0 - benchmark_gross,
                    "benchmark_substitution50_bps": stock_gross - benchmark_gross - 50.0,
                }
                fold_records.append(record)
                symbol_records.append(record)
                all_records.append(record)
            fold_rows.append({
                "fold": fold,
                "primary_states": len(fold_records),
                "stock_net25_mean_bps": _mean([r["stock_net25_bps"] for r in fold_records]),
                "stock_after25_minus_benchmark_mean_bps": _mean(
                    [r["stock_after25_minus_benchmark_gross_bps"] for r in fold_records]
                ),
                "benchmark_substitution50_mean_bps": _mean(
                    [r["benchmark_substitution50_bps"] for r in fold_records]
                ),
            })

        net_mean = _mean([r["stock_net25_bps"] for r in symbol_records])
        alpha_mean = _mean([r["stock_after25_minus_benchmark_gross_bps"] for r in symbol_records])
        substitution_mean = _mean([r["benchmark_substitution50_bps"] for r in symbol_records])
        positive_net_folds = sum(
            1 for row in fold_rows
            if row["stock_net25_mean_bps"] is not None and row["stock_net25_mean_bps"] > 0
        )
        positive_alpha_folds = sum(
            1 for row in fold_rows
            if row["stock_after25_minus_benchmark_mean_bps"] is not None
            and row["stock_after25_minus_benchmark_mean_bps"] > 0
        )
        positive_substitution_folds = sum(
            1 for row in fold_rows
            if row["benchmark_substitution50_mean_bps"] is not None
            and row["benchmark_substitution50_mean_bps"] > 0
        )
        alpha_pass = bool(
            len(symbol_records) >= 20
            and net_mean is not None and net_mean > 0
            and alpha_mean is not None and alpha_mean > 0
            and positive_net_folds >= 3
            and positive_alpha_folds >= 3
        )
        substitution_pass = bool(
            len(symbol_records) >= 20
            and substitution_mean is not None and substitution_mean > 0
            and positive_substitution_folds >= 3
        )
        per_symbol.append({
            "symbol": symbol,
            "primary_states": len(symbol_records),
            "stock_net25_mean_bps": net_mean,
            "stock_after25_minus_benchmark_mean_bps": alpha_mean,
            "benchmark_substitution50_mean_bps": substitution_mean,
            "positive_stock_net_folds": positive_net_folds,
            "positive_benchmark_alpha_folds": positive_alpha_folds,
            "positive_benchmark_substitution50_folds": positive_substitution_folds,
            "external_alpha_pass": alpha_pass,
            "benchmark_substitution50_pass": substitution_pass,
            "folds": fold_rows,
        })

    alpha_passes = sum(int(row["external_alpha_pass"]) for row in per_symbol)
    substitution_passes = sum(int(row["benchmark_substitution50_pass"]) for row in per_symbol)
    aggregate_alpha = _mean([r["stock_after25_minus_benchmark_gross_bps"] for r in all_records])
    aggregate_substitution = _mean([r["benchmark_substitution50_bps"] for r in all_records])
    return {
        "selection_calendar": {
            "rows": len(calendar),
            "first": calendar[0].isoformat(),
            "last": calendar[-1].isoformat(),
            "common_cutoff": cutoff.isoformat(),
            "benchmark_used_in_selection_calendar": False,
        },
        "holdout": list(holdout),
        "benchmark": benchmark,
        "per_symbol": per_symbol,
        "broad_external_alpha": {
            "passing_symbols": alpha_passes,
            "required": 3,
            "aggregate_stock_after25_minus_benchmark_mean_bps": aggregate_alpha,
            "passes": bool(alpha_passes >= 3 and aggregate_alpha is not None and aggregate_alpha > 0),
        },
        "broad_benchmark_substitution50": {
            "passing_symbols": substitution_passes,
            "required": 3,
            "aggregate_benchmark_substitution50_mean_bps": aggregate_substitution,
            "passes": bool(
                substitution_passes >= 3
                and aggregate_substitution is not None
                and aggregate_substitution > 0
            ),
        },
        "development_symbols_loaded": False,
        "holdout_target_rows_used_in_training": False,
    }


def _assert_homebuilder_parity(result: dict, expected: dict) -> dict:
    alpha = result["broad_external_alpha"]
    substitution = result["broad_benchmark_substitution50"]
    checks = {
        "external_alpha_passing_symbols": alpha["passing_symbols"] == expected["external_alpha_passing_symbols"],
        "external_alpha_passes": alpha["passes"] is expected["external_alpha_passes"],
        "aggregate_alpha_rounded2": round(alpha["aggregate_stock_after25_minus_benchmark_mean_bps"], 2)
        == expected["aggregate_matched_benchmark_alpha_bps_rounded2"],
        "substitution50_passing_symbols": substitution["passing_symbols"] == expected["substitution50_passing_symbols"],
        "substitution50_passes": substitution["passes"] is expected["substitution50_passes"],
        "aggregate_substitution50_rounded2": round(substitution["aggregate_benchmark_substitution50_mean_bps"], 2)
        == expected["aggregate_substitution50_bps_rounded2"],
    }
    by_symbol = {row["symbol"]: row for row in result["per_symbol"]}
    checks["symbol_alpha_pass"] = all(
        bool(by_symbol[symbol]["external_alpha_pass"]) is bool(passed)
        for symbol, passed in expected["symbol_alpha_pass"].items()
    )
    checks["symbol_substitution50_pass"] = all(
        bool(by_symbol[symbol]["benchmark_substitution50_pass"]) is bool(passed)
        for symbol, passed in expected["symbol_substitution50_pass"].items()
    )
    if not all(checks.values()):
        raise RuntimeError(f"HOME_BUILDER_PARITY_FAILED {json.dumps(checks, sort_keys=True)}")
    return checks


def run(contract_path: Path, output_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    generic = contract["generic_prior"]
    if tuple(generic["training_universe"]) != TRAIN:
        raise RuntimeError("training universe drift")
    if list(generic["features"]) != FEATURES:
        raise RuntimeError("feature drift")
    if int(generic["delay_sessions"]) != DELAY or int(generic["hold_sessions"]) != HOLD:
        raise RuntimeError("execution horizon drift")
    if float(generic["cost_bps"]) != COST_BPS:
        raise RuntimeError("cost drift")
    if generic["ticker_identity"] is not False or generic["holdout_target_rows_used_in_training"] is not False:
        raise RuntimeError("target-exclusion boundary drift")

    parity_contract = contract["parity_control"]
    parity_result = _evaluate_family(
        tuple(parity_contract["holdout"]),
        str(parity_contract["benchmark"]),
    )
    parity_checks = _assert_homebuilder_parity(parity_result, parity_contract["accepted_result"])

    # Bank holdouts are not loaded until the accepted Homebuilder result has reproduced.
    target = contract["target"]
    bank_result = _evaluate_family(tuple(target["holdout"]), str(target["benchmark"]))
    output = {
        "schema": "foundry.research.public_runner_industry_holdout_result.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_authority": contract["scientific_authority"],
        "parity": {
            "passed": True,
            "checks": parity_checks,
            "control_family": parity_contract["family"],
            "result": parity_result,
        },
        "target": {
            "family": target["family"],
            "result": bank_result,
        },
        "boundaries": {
            **contract["boundaries"],
            "homebuilder_parity_required_before_bank_load": True,
            "homebuilder_parity_passed": True,
            "bank_holdout_target_rows_used_in_training": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.contract), Path(args.output))


if __name__ == "__main__":
    main()
