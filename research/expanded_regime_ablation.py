from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOTS = ["CL", "GC", "MNQ", "NQ", "ZN"]
SOURCE_COMMIT = "60abd3fb6369c6ce0b6a4a65b0f2562fc96b1264"

BASE_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "range_frac", "body_frac",
    "log_volume", "volume_z_24", "volume_z_120", "rv_12", "rv_60",
    "ema12_dist", "ema48_dist", "mom_24", "mom_120", "minute_density",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

EXPANDED_FEATURES = [
    # multi-horizon returns / momentum
    "ret_24", "ret_48", "ret_120", "roc_5", "roc_20", "roc_60",
    # candle / path shape
    "upper_wick_frac", "lower_wick_frac", "close_location", "gap_frac",
    # volume / flow
    "volume_z_60", "volume_change_1", "mfi_14", "mfi_28", "vwap_dist_pct",
    # volatility
    "rv_6", "rv_24", "rv_120", "vol_ratio_12_120", "atr14_pct", "atr28_pct",
    "true_range_pct", "range_z_60",
    # trend / structure
    "ema9_dist", "ema20_dist", "ema50_dist", "ema100_dist", "ema200_dist",
    "ema9_20_spread", "ema20_50_spread", "ema50_200_spread",
    "trend_slope_20", "trend_slope_60", "efficiency_10", "efficiency_30",
    # oscillator / statistics
    "rsi_7", "rsi_14", "rsi_28", "z_close_20", "z_close_60", "z_close_120",
    "z_volume_20", "z_volume_60", "z_volume_120", "ret_skew_20", "ret_skew_60",
    "ret_kurt_20", "ret_kurt_60", "ret_autocorr_20", "ret_autocorr_60",
    # bands / breakout position
    "bb20_width", "bb20_pos", "bb50_width", "bb50_pos",
    "donchian20_pos", "donchian55_pos",
]

REGIME_FEATURES = [
    "vol_regime_low", "vol_regime_mid", "vol_regime_high",
    "trend_regime_down", "trend_regime_mixed", "trend_regime_up",
    "liquidity_regime_low", "liquidity_regime_mid", "liquidity_regime_high",
    "session_asia", "session_europe", "session_us",
    "stress_regime",
]

CONTEXT_BASE = ["ret_1", "ret_6", "rv_12", "atr14_pct", "z_close_20"]


def _zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=0).replace(0, np.nan)
    return (s - mean) / std


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    gain = _rma(d.clip(lower=0), n)
    loss = _rma((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    out = out.mask((loss == 0) & (gain == 0), 50.0)
    return out


def _mfi(frame: pd.DataFrame, n: int) -> pd.Series:
    tp = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    raw = tp * frame["volume"].clip(lower=0)
    d = tp.diff()
    pos = raw.where(d > 0, 0.0).rolling(n, min_periods=n).sum()
    neg = raw.where(d < 0, 0.0).rolling(n, min_periods=n).sum()
    ratio = pos / neg.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + ratio)
    out = out.mask((neg == 0) & (pos > 0), 100.0)
    out = out.mask((neg == 0) & (pos == 0), 50.0)
    return out


def _efficiency(close: pd.Series, n: int) -> pd.Series:
    change = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum().replace(0, np.nan)
    return change / path


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def _build_bars(path: Path, root: str) -> pd.DataFrame:
    f = pd.read_csv(path)
    expected = ["datetime", "open", "high", "low", "close", "volume"]
    if list(f.columns) != expected:
        raise RuntimeError(f"{root}: unexpected schema {list(f.columns)}")
    f["timestamp"] = pd.to_datetime(f["datetime"], utc=True, errors="raise")
    for c in ["open", "high", "low", "close", "volume"]:
        f[c] = pd.to_numeric(f[c], errors="raise")
    f = f.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
    w = f.set_index("timestamp")
    b = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), observed_minutes=("close", "count")
    )
    b = b[b["observed_minutes"] > 0].reset_index()
    b["market"] = root
    return b


def _add_features(b: pd.DataFrame) -> pd.DataFrame:
    g = b.copy()
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    open_ = g["open"].astype(float)
    vol = g["volume"].astype(float).clip(lower=0)
    log_close = np.log(close)
    ret1 = log_close.diff()

    for lag in (1, 3, 6, 12, 24, 48, 120):
        g[f"ret_{lag}"] = log_close.diff(lag)
    g["roc_5"] = close.pct_change(5)
    g["roc_20"] = close.pct_change(20)
    g["roc_60"] = close.pct_change(60)
    g["mom_24"] = close.pct_change(24)
    g["mom_120"] = close.pct_change(120)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    g["range_frac"] = (high - low) / close.replace(0, np.nan)
    g["body_frac"] = (close - open_) / open_.replace(0, np.nan)
    g["upper_wick_frac"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close.replace(0, np.nan)
    g["lower_wick_frac"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close.replace(0, np.nan)
    g["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    g["gap_frac"] = (open_ - prev_close) / prev_close.replace(0, np.nan)
    g["true_range_pct"] = tr / close.replace(0, np.nan)
    g["range_z_60"] = _zscore(g["range_frac"], 60)

    g["log_volume"] = np.log1p(vol)
    for n in (20, 24, 60, 120):
        g[f"volume_z_{n}"] = _zscore(g["log_volume"], n)
    g["volume_change_1"] = g["log_volume"].diff()

    for n in (6, 12, 24, 60, 120):
        g[f"rv_{n}"] = ret1.rolling(n, min_periods=n).std(ddof=0)
    g["vol_ratio_12_120"] = g["rv_12"] / g["rv_120"].replace(0, np.nan)
    g["atr14_pct"] = _rma(tr, 14) / close.replace(0, np.nan)
    g["atr28_pct"] = _rma(tr, 28) / close.replace(0, np.nan)

    ema: dict[int, pd.Series] = {}
    for n in (9, 12, 20, 48, 50, 100, 200):
        ema[n] = close.ewm(span=n, adjust=False, min_periods=n).mean()
    g["ema12_dist"] = close / ema[12] - 1.0
    g["ema48_dist"] = close / ema[48] - 1.0
    for n in (9, 20, 50, 100, 200):
        g[f"ema{n}_dist"] = close / ema[n] - 1.0
    g["ema9_20_spread"] = ema[9] / ema[20] - 1.0
    g["ema20_50_spread"] = ema[20] / ema[50] - 1.0
    g["ema50_200_spread"] = ema[50] / ema[200] - 1.0
    g["trend_slope_20"] = close.pct_change(20) / 20.0
    g["trend_slope_60"] = close.pct_change(60) / 60.0
    g["efficiency_10"] = _efficiency(close, 10)
    g["efficiency_30"] = _efficiency(close, 30)

    for n in (7, 14, 28):
        g[f"rsi_{n}"] = _rsi(close, n)
    g["mfi_14"] = _mfi(g, 14)
    g["mfi_28"] = _mfi(g, 28)
    for n in (20, 60, 120):
        g[f"z_close_{n}"] = _zscore(close, n)
        g[f"z_volume_{n}"] = _zscore(g["log_volume"], n)
    for n in (20, 60):
        g[f"ret_skew_{n}"] = ret1.rolling(n, min_periods=n).skew()
        g[f"ret_kurt_{n}"] = ret1.rolling(n, min_periods=n).kurt()
        g[f"ret_autocorr_{n}"] = ret1.rolling(n, min_periods=n).corr(ret1.shift(1))

    for n in (20, 50):
        mean = close.rolling(n, min_periods=n).mean()
        std = close.rolling(n, min_periods=n).std(ddof=0)
        upper = mean + 2.0 * std
        lower = mean - 2.0 * std
        g[f"bb{n}_width"] = (upper - lower) / mean.abs().replace(0, np.nan)
        g[f"bb{n}_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)
    for n in (20, 55):
        hi = high.rolling(n, min_periods=n).max()
        lo = low.rolling(n, min_periods=n).min()
        g[f"donchian{n}_pos"] = (close - lo) / (hi - lo).replace(0, np.nan)

    session = g["timestamp"].dt.date
    tp = (high + low + close) / 3.0
    cv = vol.groupby(session).cumsum()
    cpv = (tp * vol).groupby(session).cumsum()
    vwap = (cpv / cv).where(cv > 0)
    g["vwap_dist_pct"] = (close - vwap) / vwap.replace(0, np.nan)

    g["minute_density"] = g["observed_minutes"].clip(upper=12) / 12.0
    hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
    g["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    g["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    dow = g["timestamp"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * math.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * math.pi * dow / 7.0)

    # Fixed, causal regime encodings. No full-sample quantiles are used.
    vr = g["vol_ratio_12_120"]
    g["vol_regime_low"] = (vr < 0.8).astype(float)
    g["vol_regime_mid"] = ((vr >= 0.8) & (vr <= 1.25)).astype(float)
    g["vol_regime_high"] = (vr > 1.25).astype(float)
    down = (ema[20] < ema[50]) & (ema[50] < ema[200])
    up = (ema[20] > ema[50]) & (ema[50] > ema[200])
    g["trend_regime_down"] = down.astype(float)
    g["trend_regime_up"] = up.astype(float)
    g["trend_regime_mixed"] = (~(down | up)).astype(float)
    vz = g["z_volume_120"]
    g["liquidity_regime_low"] = (vz < -0.75).astype(float)
    g["liquidity_regime_mid"] = ((vz >= -0.75) & (vz <= 0.75)).astype(float)
    g["liquidity_regime_high"] = (vz > 0.75).astype(float)
    g["session_asia"] = ((hour >= 0) & (hour < 7)).astype(float)
    g["session_europe"] = ((hour >= 7) & (hour < 13)).astype(float)
    g["session_us"] = ((hour >= 13) | (hour < 0)).astype(float)
    g["stress_regime"] = ((g["vol_ratio_12_120"] > 1.5) & (g["range_z_60"] > 1.0)).astype(float)

    for h in (3, 6, 12, 24):
        fwd = close.shift(-h) / close - 1.0
        g[f"target_dir_h{h}"] = np.where(fwd.notna(), (fwd > 0).astype(int), np.nan)
        for mult, label in ((0.25, "025"), (0.50, "050")):
            th = mult * g["atr14_pct"]
            cls = np.where(fwd > th, 2, np.where(fwd < -th, 0, 1))
            g[f"target_move_h{h}_k{label}"] = np.where(fwd.notna() & th.notna(), cls, np.nan)
    return g.replace([np.inf, -np.inf], np.nan)


def _contextualize(frames: dict[str, pd.DataFrame]) -> list[str]:
    context_cols: list[str] = []
    for target_root, target in frames.items():
        idx = pd.DatetimeIndex(target["timestamp"])
        for source_root, source in frames.items():
            s = source.set_index("timestamp")
            for base in CONTEXT_BASE:
                name = f"ctx_{source_root}_{base}"
                aligned = s[base].reindex(idx).ffill(limit=1)
                target[name] = aligned.to_numpy()
                if name not in context_cols:
                    context_cols.append(name)
    return context_cols


def _row_folds(n: int, horizon: int) -> list[tuple[int, int, int, int]]:
    first = n // 2
    size = (n - first) // 4
    if size < 250:
        raise RuntimeError(f"insufficient OOS fold size {size}")
    out = []
    for i in range(4):
        ts = first + i * size
        te = n if i == 3 else first + (i + 1) * size
        tr_end = ts - horizon
        if tr_end < 750:
            raise RuntimeError("insufficient purged training rows")
        out.append((0, tr_end, ts, te))
    return out


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
    ]).fit(X, y)


def _evaluate(frame: pd.DataFrame, features: list[str], target: str, horizon: int) -> dict | None:
    work = frame[[*features, target]].dropna().reset_index(drop=True)
    if len(work) < 2200:
        return None
    X = work[features].to_numpy(float)
    y = work[target].astype(int).to_numpy()
    try:
        folds = _row_folds(len(work), horizon)
    except RuntimeError:
        return None
    fold_rows = []
    for i, (s, e, ts, te) in enumerate(folds):
        if len(np.unique(y[s:e])) < 2:
            return None
        model = _fit_logistic(X[s:e], y[s:e])
        pred = model.predict(X[ts:te])
        fold_rows.append({"fold": i, "train_rows": e - s, "test_rows": te - ts, **_metric(y[ts:te], pred)})
    disc = [r["balanced_accuracy"] for r in fold_rows[:3]]
    return {
        "rows": int(len(work)),
        "class_count": int(len(np.unique(y))),
        "discovery_ba_mean": float(np.mean(disc)),
        "discovery_ba_std": float(np.std(disc)),
        "discovery_ba_floor": float(np.min(disc)),
        "selection_score": float(np.mean(disc) - 0.5 * np.std(disc)),
        "holdout": fold_rows[3],
        "folds": fold_rows,
    }


def _challengers(frame: pd.DataFrame, features: list[str], target: str, horizon: int) -> dict:
    work = frame[[*features, target]].dropna().reset_index(drop=True)
    X = work[features].to_numpy(float)
    y = work[target].astype(int).to_numpy()
    folds = _row_folds(len(work), horizon)
    _, train_end, test_start, test_end = folds[3]
    out = {}
    factories = {
        "logistic_regression": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
        ]),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=240, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }
    for name, factory in factories.items():
        m = factory().fit(X[:train_end], y[:train_end])
        pred = m.predict(X[test_start:test_end])
        out[name] = _metric(y[test_start:test_end], pred)
    return {"train_rows": int(train_end), "holdout_rows": int(test_end - test_start), "models": out}


def _target_specs() -> list[tuple[str, int]]:
    out = []
    for h in (3, 6, 12, 24):
        out.append((f"target_dir_h{h}", h))
        out.append((f"target_move_h{h}_k025", h))
        out.append((f"target_move_h{h}_k050", h))
    return out


def _feature_modes(context_cols: list[str]) -> dict[str, list[str]]:
    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES))
    with_regime = list(dict.fromkeys(expanded + REGIME_FEATURES))
    with_context = list(dict.fromkeys(with_regime + context_cols))
    return {
        "baseline20": BASE_FEATURES,
        "expanded": expanded,
        "expanded_regime": with_regime,
        "expanded_regime_context": with_context,
    }


def _pooled_frame(frames: dict[str, pd.DataFrame], market_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    pieces = []
    id_cols = [f"market_{r}" for r in ROOTS]
    targets = [name for name, _ in _target_specs()]
    for root, f in frames.items():
        p = f[["timestamp", *market_features, *targets]].copy()
        p["market"] = root
        for r in ROOTS:
            p[f"market_{r}"] = float(r == root)
        pieces.append(p)
    pooled = pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "market"]).reset_index(drop=True)
    return pooled, id_cols


def _time_folds(frame: pd.DataFrame, horizon: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    times = pd.Series(pd.to_datetime(frame["timestamp"], utc=True).dropna().unique()).sort_values().reset_index(drop=True)
    first = len(times) // 2
    size = (len(times) - first) // 4
    if size < 250:
        raise RuntimeError("insufficient pooled time fold")
    folds = []
    purge = pd.Timedelta(minutes=12 * horizon)
    for i in range(4):
        start = times.iloc[first + i * size]
        end = times.iloc[-1] + pd.Timedelta(microseconds=1) if i == 3 else times.iloc[first + (i + 1) * size]
        train_cut = start - purge
        folds.append((train_cut, start, end))
    return folds


def _evaluate_pooled(frame: pd.DataFrame, features: list[str], target: str, horizon: int) -> dict | None:
    work = frame[["timestamp", *features, target]].dropna().sort_values("timestamp").reset_index(drop=True)
    if len(work) < 8000:
        return None
    folds = _time_folds(work, horizon)
    fold_rows = []
    for i, (train_cut, start, end) in enumerate(folds):
        train = work[work["timestamp"] < train_cut]
        test = work[(work["timestamp"] >= start) & (work["timestamp"] < end)]
        if len(train) < 3000 or len(test) < 800:
            return None
        ytr = train[target].astype(int).to_numpy(); yte = test[target].astype(int).to_numpy()
        if len(np.unique(ytr)) < 2:
            return None
        m = _fit_logistic(train[features].to_numpy(float), ytr)
        pred = m.predict(test[features].to_numpy(float))
        fold_rows.append({"fold": i, "train_rows": int(len(train)), "test_rows": int(len(test)), **_metric(yte, pred)})
    disc = [r["balanced_accuracy"] for r in fold_rows[:3]]
    return {
        "rows": int(len(work)), "class_count": int(work[target].nunique()),
        "discovery_ba_mean": float(np.mean(disc)), "discovery_ba_std": float(np.std(disc)),
        "discovery_ba_floor": float(np.min(disc)),
        "selection_score": float(np.mean(disc) - 0.5 * np.std(disc)),
        "holdout": fold_rows[3], "folds": fold_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    frames: dict[str, pd.DataFrame] = {}
    source_evidence = {}
    for root in ROOTS:
        matches = list((args.source_root / root).glob(f"{root}_1min_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"{root}: expected one pinned 1min file, got {matches}")
        bars = _build_bars(matches[0], root)
        frames[root] = _add_features(bars)
        source_evidence[root] = {
            "file": matches[0].name, "source_rows": int(pd.read_csv(matches[0], usecols=["datetime"]).shape[0]),
            "bars_12min": int(len(bars)), "first": bars["timestamp"].iloc[0].isoformat(), "last": bars["timestamp"].iloc[-1].isoformat(),
        }

    context_cols = _contextualize(frames)
    modes = _feature_modes(context_cols)
    target_specs = _target_specs()

    result = {
        "schema": "foundry.expanded_regime_ablation.v1",
        "research_only": True,
        "promotion_authority": False,
        "source_repo": "axb0306/cme-futures-ohlc",
        "source_commit": SOURCE_COMMIT,
        "causal_admission": {
            "excluded_forward_aligned_features": ["chikou_span"],
            "note": "all engineered features use information available at or before each completed 12Min bar",
        },
        "feature_modes": {k: len(v) for k, v in modes.items()},
        "target_specs": [name for name, _ in target_specs],
        "source_evidence": source_evidence,
        "markets": {},
        "pooled": {},
    }

    for root, frame in frames.items():
        candidates = []
        current_target_modes = {}
        for mode, features in modes.items():
            for target, horizon in target_specs:
                ev = _evaluate(frame, features, target, horizon)
                if ev is None:
                    continue
                row = {"feature_mode": mode, "target": target, "horizon_rows": horizon, **ev}
                candidates.append(row)
                if target == "target_dir_h12":
                    current_target_modes[mode] = ev
        candidates.sort(key=lambda x: (x["selection_score"], x["discovery_ba_floor"]), reverse=True)
        if not candidates:
            raise RuntimeError(f"{root}: no evaluable candidates")
        best = candidates[0]
        challenge = _challengers(frame, modes[best["feature_mode"]], best["target"], int(best["horizon_rows"]))
        result["markets"][root] = {
            "current_target_mode_comparison": current_target_modes,
            "best_discovery_candidate": best,
            "best_holdout_challengers": challenge,
            "top10": candidates[:10],
        }
        print(root, best["feature_mode"], best["target"], "discovery", round(best["discovery_ba_mean"], 6), "holdout", round(best["holdout"]["balanced_accuracy"], 6))

    pooled_market_features = modes["expanded_regime_context"]
    pooled, id_cols = _pooled_frame(frames, pooled_market_features)
    pooled_modes = {
        "baseline20_market_id": BASE_FEATURES + id_cols,
        "expanded_market_id": list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + id_cols)),
        "expanded_regime_market_id": list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES + id_cols)),
        "expanded_regime_context_market_id": list(dict.fromkeys(pooled_market_features + id_cols)),
    }
    pooled_candidates = []
    for mode, features in pooled_modes.items():
        for target, horizon in target_specs:
            ev = _evaluate_pooled(pooled, features, target, horizon)
            if ev is not None:
                pooled_candidates.append({"feature_mode": mode, "target": target, "horizon_rows": horizon, **ev})
    pooled_candidates.sort(key=lambda x: (x["selection_score"], x["discovery_ba_floor"]), reverse=True)
    if pooled_candidates:
        result["pooled"] = {
            "feature_modes": {k: len(v) for k, v in pooled_modes.items()},
            "best_discovery_candidate": pooled_candidates[0],
            "top10": pooled_candidates[:10],
        }
        pbest = pooled_candidates[0]
        print("POOLED", pbest["feature_mode"], pbest["target"], "discovery", round(pbest["discovery_ba_mean"], 6), "holdout", round(pbest["holdout"]["balanced_accuracy"], 6))

    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("EXPANDED_REGIME_ABLATION=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
