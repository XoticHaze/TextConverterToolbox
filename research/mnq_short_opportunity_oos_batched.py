from __future__ import annotations

"""Semantics-preserving batched evaluator for the offline MNQ short specialist.

The original evaluator recomputes the full eligible-training mask and invokes
predict_proba once per row even though models change only at the frozen refit
cadence. This module preserves the exact training/refit contract while batching
predictions between refits.
"""

import numpy as np
import pandas as pd

from research.mnq_short_opportunity_oos import ShortOOSSpec, challenger_factories
from research.mnq_short_opportunity_targets import assert_short_feature_fence


def _eligible_prefix_receipts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return original-evaluator eligible row count/max-resolution for every decision row."""
    resolution = pd.to_numeric(frame["target_resolution_row"], errors="coerce").to_numpy(float)
    labelled = frame["short_label"].notna().to_numpy(bool) & np.isfinite(resolution)
    n = len(frame)

    starts: list[list[int]] = [[] for _ in range(n + 1)]
    for row in np.flatnonzero(labelled):
        first_eligible = int(resolution[row]) + 1
        if first_eligible < 0:
            first_eligible = 0
        if first_eligible <= n:
            starts[first_eligible].append(int(row))

    count = np.zeros(n, dtype=np.int64)
    max_resolution = np.full(n, -1, dtype=np.int64)
    running_count = 0
    running_max = -1
    for i in range(n):
        for row in starts[i]:
            running_count += 1
            running_max = max(running_max, int(resolution[row]))
        count[i] = running_count
        max_resolution[i] = running_max
    return count, max_resolution


def _training_indices(frame: pd.DataFrame, decision_row: int) -> np.ndarray:
    resolution = pd.to_numeric(frame["target_resolution_row"], errors="coerce")
    eligible = frame["short_label"].notna() & resolution.notna() & (resolution < decision_row)
    return np.flatnonzero(eligible.to_numpy())


def evaluate_short_challengers_batched(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: ShortOOSSpec = ShortOOSSpec(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce the same OOS stream as the original evaluator with batched inference.

    Models are fit at exactly the same first eligible decision row and then every
    ``retrain_every`` decision rows. The training set at each fit is still defined
    strictly by ``target_resolution_row < decision_row``. Predictions between fits
    are evaluated in one batch, while the emitted train-row/max-resolution receipts
    retain the original evaluator's per-decision-row values.
    """
    assert_short_feature_fence(feature_columns)
    required = set(feature_columns) | {"short_label", "short_forward_points", "target_resolution_row"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame must be chronological")

    x = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    n = len(frame)
    eligible_count, eligible_max_resolution = _eligible_prefix_receipts(frame)

    first_fit: int | None = None
    for i in range(n):
        if eligible_count[i] < spec.min_train_rows:
            continue
        train_idx = _training_indices(frame, i)
        if len(train_idx) < spec.min_train_rows:
            continue
        if frame.iloc[train_idx]["short_label"].astype(int).nunique() < 2:
            continue
        first_fit = i
        break

    if first_fit is None:
        return pd.DataFrame(), pd.DataFrame()

    predictions: list[dict[str, object]] = []
    fit_at = first_fit
    while fit_at < n:
        train_idx = _training_indices(frame, fit_at)
        if len(train_idx) < spec.min_train_rows:
            raise RuntimeError("batched evaluator lost frozen minimum training support")
        y_train = frame.iloc[train_idx]["short_label"].astype(int)
        if y_train.nunique() < 2:
            raise RuntimeError("batched evaluator lost two-class training support after first fit")

        models: dict[str, object] = {}
        for name, factory in challenger_factories().items():
            model = factory()
            model.fit(x.iloc[train_idx], y_train)
            models[name] = model

        block_end = min(n, fit_at + spec.retrain_every)
        block_rows = np.arange(fit_at, block_end, dtype=np.int64)
        valid = frame.iloc[block_rows]["short_forward_points"].notna().to_numpy(bool)
        score_rows = block_rows[valid]
        if len(score_rows):
            row_x = x.iloc[score_rows]
            gross_all = frame.iloc[score_rows]["short_forward_points"].astype(float).to_numpy()
            for name, model in models.items():
                probabilities = np.asarray(model.predict_proba(row_x)[:, 1], dtype=float)
                for pos, row in enumerate(score_rows):
                    p_short = float(probabilities[pos])
                    participate = p_short >= spec.probability_threshold
                    gross = float(gross_all[pos]) if participate else 0.0
                    net = gross - spec.point_cost if participate else 0.0
                    predictions.append({
                        "row": int(row),
                        "index": frame.index[int(row)],
                        "challenger": name,
                        "p_short": p_short,
                        "participate": participate,
                        "gross_points": gross,
                        "net_points": net,
                        "train_rows": int(eligible_count[int(row)]),
                        "max_train_resolution_row": int(eligible_max_resolution[int(row)]),
                    })

        fit_at += spec.retrain_every

    pred = pd.DataFrame(predictions)
    if pred.empty:
        return pred, pd.DataFrame()
    pred = pred.sort_values(["row", "challenger"], kind="stable").reset_index(drop=True)
    summary = pred.groupby("challenger", as_index=False).agg(
        oos_rows=("row", "size"),
        signals=("participate", "sum"),
        gross_points=("gross_points", "sum"),
        net_points=("net_points", "sum"),
        mean_net_points=("net_points", "mean"),
    )
    summary["coverage"] = summary["signals"] / summary["oos_rows"]
    return pred, summary
