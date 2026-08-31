from __future__ import annotations

"""Chronological OOS evaluator for the offline MNQ short-opportunity specialist.

Research only. Produces historical challenger evidence; it is not short-execution,
broker, StrategySpec, or promotion authority.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.mnq_short_opportunity_targets import assert_short_feature_fence


@dataclass(frozen=True)
class ShortOOSSpec:
    min_train_rows: int = 500
    retrain_every: int = 250
    probability_threshold: float = 0.55
    point_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.min_train_rows < 20 or self.retrain_every < 1:
            raise ValueError("invalid training cadence")
        if not 0.5 <= self.probability_threshold < 1.0:
            raise ValueError("probability_threshold must be in [0.5, 1)")
        if self.point_cost < 0:
            raise ValueError("point_cost must be non-negative")


def challenger_factories(random_state: int = 17) -> dict[str, Callable[[], object]]:
    return {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        ),
        "hist_gb": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=random_state),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=8, class_weight="balanced", n_jobs=-1, random_state=random_state
        ),
    }


def evaluate_short_challengers(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: ShortOOSSpec = ShortOOSSpec(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate purged, past-only OOS predictions and economic summaries.

    A row is eligible for training at decision row i only when its target_resolution_row
    is strictly before i. This is the horizon purge. Models are refit on a fixed cadence
    using only those resolved historical rows. Economic scoring uses the already-frozen
    short_forward_points and charges point_cost only when the challenger participates.
    """
    assert_short_feature_fence(feature_columns)
    required = set(feature_columns) | {"short_label", "short_forward_points", "target_resolution_row"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame must be chronological")

    x = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    predictions: list[dict[str, object]] = []
    models: dict[str, object] = {}
    last_fit_at: int | None = None

    for i in range(len(frame)):
        eligible = (
            frame["short_label"].notna()
            & frame["target_resolution_row"].notna()
            & (frame["target_resolution_row"].astype(float) < i)
        )
        train_idx = np.flatnonzero(eligible.to_numpy())
        if len(train_idx) < spec.min_train_rows:
            continue
        y_train = frame.iloc[train_idx]["short_label"].astype(int)
        if y_train.nunique() < 2:
            continue
        if last_fit_at is None or i - last_fit_at >= spec.retrain_every:
            models = {}
            for name, factory in challenger_factories().items():
                model = factory()
                model.fit(x.iloc[train_idx], y_train)
                models[name] = model
            last_fit_at = i
        if pd.isna(frame.iloc[i]["short_forward_points"]) or not models:
            continue
        row_x = x.iloc[[i]]
        for name, model in models.items():
            p_short = float(model.predict_proba(row_x)[0, 1])
            participate = p_short >= spec.probability_threshold
            gross = float(frame.iloc[i]["short_forward_points"]) if participate else 0.0
            net = gross - spec.point_cost if participate else 0.0
            predictions.append({
                "row": i,
                "index": frame.index[i],
                "challenger": name,
                "p_short": p_short,
                "participate": participate,
                "gross_points": gross,
                "net_points": net,
                "train_rows": len(train_idx),
                "max_train_resolution_row": int(frame.iloc[train_idx]["target_resolution_row"].max()),
            })

    pred = pd.DataFrame(predictions)
    if pred.empty:
        return pred, pd.DataFrame()
    summary = pred.groupby("challenger", as_index=False).agg(
        oos_rows=("row", "size"),
        signals=("participate", "sum"),
        gross_points=("gross_points", "sum"),
        net_points=("net_points", "sum"),
        mean_net_points=("net_points", "mean"),
    )
    summary["coverage"] = summary["signals"] / summary["oos_rows"]
    return pred, summary
