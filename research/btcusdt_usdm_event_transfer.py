from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, EXPANDED_FEATURES, REGIME_FEATURES, _add_features
from research.mnq_nonoverlap_phase_audit import BAR_NS, phases
from research.mnq_opportunity_target_matrix import model
from research.mnq_triple_barrier_events import build_events, classification, realized_returns, summarize

CONFIGS = {
    "h24_bar05": (24, 0.5),
    "h24_bar10": (24, 1.0),
}
BINANCE_REFERENCE_COMMIT = "5c7f3197591c0d54d85dc43066226bc4c671d47a"
EXPECTED_START = pd.Timestamp("2020-01-01", tz="UTC")
EXPECTED_END_MIN = pd.Timestamp("2025-12-01", tz="UTC")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_binance_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"{path}: expected one CSV member, got {names}")
        raw = z.read(names[0])
    f = pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)
    if f.shape[1] < 6:
        raise RuntimeError(f"{path}: unexpected kline columns {f.shape[1]}")
    open_time = pd.to_numeric(f.iloc[:, 0], errors="coerce")
    if open_time.isna().iloc[0]:
        f = f.iloc[1:].reset_index(drop=True)
        open_time = pd.to_numeric(f.iloc[:, 0], errors="raise")
    else:
        open_time = open_time.astype("int64")
    median = float(np.median(open_time.to_numpy(float)))
    unit = "us" if median > 1e14 else "ms"
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(open_time.astype("int64"), unit=unit, utc=True),
        "open": pd.to_numeric(f.iloc[:, 1], errors="raise"),
        "high": pd.to_numeric(f.iloc[:, 2], errors="raise"),
        "low": pd.to_numeric(f.iloc[:, 3], errors="raise"),
        "close": pd.to_numeric(f.iloc[:, 4], errors="raise"),
        "volume": pd.to_numeric(f.iloc[:, 5], errors="raise"),
    })
    return out


def load_monthlies(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    files = sorted(root.rglob("BTCUSDT-1m-*.zip"))
    if len(files) < 60:
        raise RuntimeError(f"insufficient BTCUSDT monthly archives: {len(files)}")
    hashes = {}
    parts = []
    for path in files:
        hashes[path.name] = sha256_file(path)
        parts.append(read_binance_zip(path))
    f = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    f = f.drop_duplicates("timestamp", keep=False).reset_index(drop=True)
    if f.empty or f["timestamp"].iloc[0] > EXPECTED_START or f["timestamp"].iloc[-1] < EXPECTED_END_MIN:
        raise RuntimeError(f"unexpected BTC coverage {f['timestamp'].iloc[0]} -> {f['timestamp'].iloc[-1]}")
    return f, hashes


def bars_12m(minutes: pd.DataFrame) -> pd.DataFrame:
    w = minutes.set_index("timestamp")
    b = w.resample("12min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), observed_minutes=("close", "count"),
    )
    b = b[b["observed_minutes"] > 0].reset_index()
    b["market"] = "BTCUSDT_PERP"
    return b


def outer_quarters() -> list[pd.Timestamp]:
    return list(pd.date_range("2022-01-01", "2026-01-01", freq="QS", tz="UTC"))


def evaluate_config(work: pd.DataFrame, horizon: int, mult: float, features: list[str]) -> dict:
    phase_nodes = {}
    for phase in phases(horizon):
        events = build_events(work, horizon, mult, phase, features)
        returns: list[float] = []
        quarter_net: list[float] = []
        quarter_rows = []
        all_y: list[int] = []
        all_pred: list[int] = []
        for start, end in zip(outer_quarters()[:-1], outer_quarters()[1:]):
            train = events[(events["timestamp"] < start) & (events["event_end_timestamp"] < start)]
            test = events[(events["timestamp"] >= start) & (events["timestamp"] < end) & (events["event_end_timestamp"] < end)]
            if len(train) < 500 or len(test) < 30:
                continue
            y_train = train["target"].astype(int).to_numpy()
            y_test = test["target"].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2:
                continue
            fitted = model().fit(train[features].to_numpy(float), y_train)
            pred = fitted.predict(test[features].to_numpy(float)).astype(int)
            r = realized_returns(test, pred)
            qmean = float(np.mean(r)) if len(r) else None
            if len(r):
                returns.extend(r.tolist())
                quarter_net.append(qmean)
            all_y.extend(y_test.tolist())
            all_pred.extend(pred.tolist())
            quarter_rows.append({
                "period": f"{start.year}Q{((start.month - 1)//3)+1}",
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "directional_signals": int(len(r)),
                "mean_net_after_2bp": qmean,
                "classification": classification(y_test, pred),
                "observed_train_classes": sorted(int(c) for c in np.unique(y_train)),
                "observed_test_classes": sorted(int(c) for c in np.unique(y_test)),
            })
        if len(quarter_rows) < 8:
            raise RuntimeError(f"phase {phase}: insufficient outer quarters {len(quarter_rows)}")
        phase_nodes[str(phase)] = {
            "quarter_rows": quarter_rows,
            "aggregate_classification": classification(np.asarray(all_y, dtype=int), np.asarray(all_pred, dtype=int)),
            "aggregate_economic": summarize(returns, quarter_net),
        }
    return {"phases": phase_nodes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    minutes, hashes = load_monthlies(args.zip_root)
    bars = bars_12m(minutes)
    frame = _add_features(bars)
    expanded = list(dict.fromkeys(BASE_FEATURES + EXPANDED_FEATURES + REGIME_FEATURES))
    cols = list(dict.fromkeys(["timestamp", "high", "low", "close", "rv_120", *expanded]))
    work = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    work["utc_slot"] = (work["timestamp"].astype("int64").to_numpy() // BAR_NS).astype(np.int64)
    feature_sets = {"baseline20": list(BASE_FEATURES), "expanded_regime": expanded}

    results = {}
    for key, (horizon, mult) in CONFIGS.items():
        results[key] = {"horizon": horizon, "barrier_multiplier": mult, "feature_sets": {}}
        for name, features in feature_sets.items():
            results[key]["feature_sets"][name] = evaluate_config(work, horizon, mult, features)
            print(key, name, "DONE")

    archive_manifest = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "schema": "foundry.btcusdt_usdm_event_transfer.v1",
        "research_only": True,
        "promotion_authority": False,
        "source": "Binance public USD-M BTCUSDT perpetual monthly 1m klines",
        "source_reference_commit": BINANCE_REFERENCE_COMMIT,
        "source_monthly_archives": len(hashes),
        "source_archive_manifest_sha256": hashlib.sha256(archive_manifest).hexdigest(),
        "source_first_timestamp": minutes["timestamp"].iloc[0].isoformat(),
        "source_last_timestamp": minutes["timestamp"].iloc[-1].isoformat(),
        "source_minute_rows": int(len(minutes)),
        "bars_12m": int(len(bars)),
        "protocol": "external-market falsification fixed before BTC results: H24 0.5x/1.0x causal-volatility triple-barrier event architectures selected from MNQ research, four fixed non-overlap phase streams, quarterly expanding past-only refit from 2020 history, 2022-2025 outer quarters, 2bp/event sensitivity, no BTC-based target selection; single-class future quarters retained",
        "excluded_forward_aligned_features": ["chikou_span"],
        "configs": results,
    }
    material = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["receipt_sha256"] = hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("BTCUSDT_USDM_EVENT_TRANSFER=PASS")
    print("RECEIPT_SHA256=" + result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
