from __future__ import annotations

import numpy as np
import pandas as pd

from research.mnq_short_opportunity_oos import ShortOOSSpec, evaluate_short_challengers
from research.mnq_short_opportunity_oos_batched import evaluate_short_challengers_batched


def _fixture(n: int = 180) -> pd.DataFrame:
    row = np.arange(n, dtype=float)
    frame = pd.DataFrame({
        "f1": np.sin(row / 7.0) + row / 1000.0,
        "f2": np.cos(row / 11.0) - row / 2000.0,
        "short_label": ((row.astype(int) % 5) < 2).astype(int),
        "short_forward_points": np.sin(row / 5.0) * 3.0 - 0.25,
        "target_resolution_row": row + 5.0,
    })
    frame.loc[n - 5 :, ["short_label", "short_forward_points", "target_resolution_row"]] = np.nan
    return frame


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["row", "challenger"], kind="stable").reset_index(drop=True)


def test_batched_evaluator_matches_original_probability_stream_and_receipts() -> None:
    frame = _fixture()
    spec = ShortOOSSpec(
        min_train_rows=40,
        retrain_every=25,
        probability_threshold=0.55,
        point_cost=1.0,
    )
    original, original_summary = evaluate_short_challengers(frame, ["f1", "f2"], spec)
    batched, batched_summary = evaluate_short_challengers_batched(frame, ["f1", "f2"], spec)

    a = _sorted(original)
    b = _sorted(batched)
    assert list(a[["row", "challenger"]].itertuples(index=False, name=None)) == list(
        b[["row", "challenger"]].itertuples(index=False, name=None)
    )
    assert np.allclose(a["p_short"].to_numpy(float), b["p_short"].to_numpy(float), rtol=0.0, atol=1e-12)
    assert np.array_equal(a["participate"].to_numpy(bool), b["participate"].to_numpy(bool))
    assert np.allclose(a["gross_points"].to_numpy(float), b["gross_points"].to_numpy(float), rtol=0.0, atol=1e-12)
    assert np.allclose(a["net_points"].to_numpy(float), b["net_points"].to_numpy(float), rtol=0.0, atol=1e-12)
    assert np.array_equal(a["train_rows"].to_numpy(int), b["train_rows"].to_numpy(int))
    assert np.array_equal(
        a["max_train_resolution_row"].to_numpy(int),
        b["max_train_resolution_row"].to_numpy(int),
    )
    assert (b["max_train_resolution_row"].to_numpy(int) < b["row"].to_numpy(int)).all()

    sa = original_summary.sort_values("challenger").reset_index(drop=True)
    sb = batched_summary.sort_values("challenger").reset_index(drop=True)
    assert list(sa["challenger"]) == list(sb["challenger"])
    for col in ["oos_rows", "signals"]:
        assert np.array_equal(sa[col].to_numpy(), sb[col].to_numpy())
    for col in ["gross_points", "net_points", "mean_net_points", "coverage"]:
        assert np.allclose(sa[col].to_numpy(float), sb[col].to_numpy(float), rtol=0.0, atol=1e-12)
