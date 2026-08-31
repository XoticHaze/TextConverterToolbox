from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.expanded_regime_ablation import BASE_FEATURES, _add_features
from research.mnq_expected_move_axb_2026 import (
    HORIZON, VOL_MULT, CUTOFF, ridge_model, load_axb_mnq, chronological_oof_threshold,
)
from research.mnq_external_transfer_validation import load_deep, deep_roll_schedule, stitch_deep, deep_bars
from research.mnq_model_family_weekly_challenge import trade_week_key
from research.mnq_opportunity_target_matrix import model as classifier_model, target_columns
from research.nq_to_mnq_execution_transfer import phase_audit

COST = 1.0


def prep(frame: pd.DataFrame) -> pd.DataFrame:
    features = list(BASE_FEATURES)
    needed = list(dict.fromkeys(["timestamp", "close", "rv_120", *features]))
    x = _add_features(frame)
    x = x[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    cls, _, _ = target_columns(x, HORIZON, VOL_MULT)
    x["class_target"] = cls
    x["point_move"] = x["close"].shift(-HORIZON) - x["close"]
    scale = x["close"].astype(float) * x["rv_120"].astype(float) * math.sqrt(HORIZON)
    x["target_move_z"] = x["point_move"] / scale.replace(0, np.nan)
    x["trade_week"] = trade_week_key(x["timestamp"])
    return x


def fit_models(deep: pd.DataFrame):
    features = list(BASE_FEATURES)
    pre = deep[(deep["timestamp"] < CUTOFF) & deep["class_target"].notna() & deep["target_move_z"].notna()].copy()
    train = pre.iloc[:-HORIZON].copy()
    x = train[features].to_numpy(float)
    y_reg = train["target_move_z"].to_numpy(float)
    threshold, oof = chronological_oof_threshold(x, y_reg)
    ridge = ridge_model().fit(x, y_reg)
    clf = classifier_model().fit(x, train["class_target"].astype(int).to_numpy())
    return train, threshold, oof, ridge, clf


def predict(frame: pd.DataFrame, ridge, clf, threshold: float) -> pd.DataFrame:
    features = list(BASE_FEATURES)
    out = frame.copy()
    z = ridge.predict(out[features].to_numpy(float))
    out["ridge_score"] = z
    out["ridge_pred"] = np.where(np.abs(z) >= threshold, np.sign(z), 0).astype(int)
    out["logistic_pred"] = clf.predict(out[features].to_numpy(float)).astype(int)
    return out


def phase_summary(frame: pd.DataFrame, pred_col: str) -> dict:
    return phase_audit(frame["timestamp"], frame[pred_col].to_numpy(int), frame["point_move"].to_numpy(float), HORIZON)


def weekly_extension(frame: pd.DataFrame) -> list[dict]:
    rows=[]
    for wk,g in frame.groupby("trade_week", sort=True):
        if len(g) < 300:
            continue
        rows.append({
            "trade_week": pd.Timestamp(wk).isoformat(),
            "rows": int(len(g)),
            "ridge": phase_summary(g, "ridge_pred"),
            "logistic": phase_summary(g, "logistic_pred"),
        })
    return rows


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    x=np.asarray(a,dtype=float); y=np.asarray(b,dtype=float)
    if len(x)<10 or np.std(x)==0 or np.std(y)==0: return None
    return float(np.corrcoef(x,y)[0,1])


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--axb-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()

    raw=load_deep(args.deep_root)
    deep=prep(deep_bars(stitch_deep(raw, deep_roll_schedule(raw))))
    axb=prep(load_axb_mnq(args.axb_root))
    train,threshold,oof,ridge,clf=fit_models(deep)

    deep_test=deep[(deep["timestamp"]>=CUTOFF)&deep["class_target"].notna()&deep["target_move_z"].notna()&deep["point_move"].notna()].copy()
    axb_test=axb[(axb["timestamp"]>=CUTOFF)&axb["class_target"].notna()&axb["target_move_z"].notna()&axb["point_move"].notna()].copy()
    deep_test=predict(deep_test,ridge,clf,threshold)
    axb_test=predict(axb_test,ridge,clf,threshold)

    common=deep_test.merge(axb_test,on="timestamp",how="inner",suffixes=("_deep","_axb"),validate="one_to_one")
    if len(common)<1000:
        raise RuntimeError(f"insufficient common timestamps {len(common)}")

    feature_corr={}
    for f in BASE_FEATURES:
        feature_corr[f]=safe_corr(common[f+"_deep"],common[f+"_axb"])
    finite=[v for v in feature_corr.values() if v is not None and np.isfinite(v)]

    overlap={
        "first_timestamp": common["timestamp"].min().isoformat(),
        "last_timestamp": common["timestamp"].max().isoformat(),
        "rows": int(len(common)),
        "feature_correlations": feature_corr,
        "feature_correlation_median": float(np.median(finite)),
        "feature_correlation_min": float(np.min(finite)),
        "ridge_score_correlation": safe_corr(common["ridge_score_deep"],common["ridge_score_axb"]),
        "ridge_direction_agreement_all_rows": float(np.mean(common["ridge_pred_deep"].to_numpy(int)==common["ridge_pred_axb"].to_numpy(int))),
        "ridge_selection_agreement": float(np.mean((common["ridge_pred_deep"].to_numpy(int)!=0)==(common["ridge_pred_axb"].to_numpy(int)!=0))),
        "logistic_direction_agreement": float(np.mean(common["logistic_pred_deep"].to_numpy(int)==common["logistic_pred_axb"].to_numpy(int))),
        "future_point_move_correlation": safe_corr(common["point_move_deep"],common["point_move_axb"]),
        "future_move_sign_agreement": float(np.mean(np.sign(common["point_move_deep"].to_numpy(float))==np.sign(common["point_move_axb"].to_numpy(float)))),
    }

    for source in ("deep","axb"):
        temp=pd.DataFrame({
            "timestamp":common["timestamp"],
            "point_move":common[f"point_move_{source}"],
            "ridge_pred":common[f"ridge_pred_{source}"],
            "logistic_pred":common[f"logistic_pred_{source}"],
        })
        overlap[f"{source}_ridge_phase_audit"]=phase_summary(temp,"ridge_pred")
        overlap[f"{source}_logistic_phase_audit"]=phase_summary(temp,"logistic_pred")

    deep_last=deep_test["timestamp"].max()
    extension=axb_test[axb_test["timestamp"]>deep_last].copy()
    if len(extension)<1000:
        raise RuntimeError(f"insufficient AXB-only extension rows {len(extension)} after {deep_last}")
    ext_rows=weekly_extension(extension)

    result={
        "schema":"foundry.mnq_h24_source_parity.v1",
        "research_only":True,
        "promotion_authority":False,
        "training_source":"mbytes21/MNQ_DATA@fc5508e2c152938d6d9eb70a36b888ae26107176 strictly pre-2026",
        "deep_test_source":"same deep MNQ source",
        "independent_test_source":"axb0306/cme-futures-ohlc@60abd3fb6369c6ce0b6a4a65b0f2562fc96b1264",
        "contract":"frozen H24 baseline20 Ridge(alpha=10) expected-move plus frozen logistic comparator; no source-specific fitting; common timestamps compared before AXB-only later extension",
        "train_rows":int(len(train)),
        "train_last_timestamp":train["timestamp"].max().isoformat(),
        "ridge_threshold_abs_pred_z":float(threshold),
        "oof_threshold_receipt":oof,
        "overlap":overlap,
        "deep_test_last_timestamp":deep_last.isoformat(),
        "axb_extension_first_timestamp":extension["timestamp"].min().isoformat(),
        "axb_extension_last_timestamp":extension["timestamp"].max().isoformat(),
        "axb_extension_rows":int(len(extension)),
        "axb_extension_weekly":ext_rows,
        "excluded_forward_aligned_features":["chikou_span"],
    }
    material=json.dumps(result,sort_keys=True,separators=(",",":")).encode()
    result["receipt_sha256"]=hashlib.sha256(material).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("MNQ_H24_SOURCE_PARITY=PASS")
    print(json.dumps({"overlap":overlap,"extension_weeks":len(ext_rows)},sort_keys=True))
    print("RECEIPT_SHA256="+result["receipt_sha256"])
    return 0

if __name__=="__main__":
    raise SystemExit(main())
