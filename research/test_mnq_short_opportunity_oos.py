from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.mnq_short_opportunity_oos import ShortOOSSpec, evaluate_short_challengers


class ShortOpportunityOOSTest(unittest.TestCase):
    def _frame(self, n: int = 180) -> pd.DataFrame:
        t = np.arange(n, dtype=float)
        label = ((t.astype(int) % 4) < 2).astype(float)
        forward = np.where(label == 1, 3.0, -2.0)
        horizon = 5
        resolution = t + horizon
        label[-horizon:] = np.nan
        forward[-horizon:] = np.nan
        resolution[-horizon:] = np.nan
        return pd.DataFrame({
            "f1": np.sin(t / 4.0),
            "f2": (t.astype(int) % 4).astype(float),
            "short_label": label,
            "short_forward_points": forward,
            "target_resolution_row": resolution,
        })

    def test_every_prediction_uses_only_resolved_past_targets(self) -> None:
        pred, summary = evaluate_short_challengers(
            self._frame(), ["f1", "f2"],
            ShortOOSSpec(min_train_rows=30, retrain_every=20, probability_threshold=0.5, point_cost=1.0),
        )
        self.assertFalse(pred.empty)
        self.assertEqual(set(pred["challenger"]), {"logistic", "hist_gb", "extra_trees"})
        self.assertTrue((pred["max_train_resolution_row"] < pred["row"]).all())
        self.assertEqual(set(summary["challenger"]), {"logistic", "hist_gb", "extra_trees"})

    def test_future_outcome_cannot_be_feature(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_short_challengers(self._frame(), ["f1", "short_forward_points"], ShortOOSSpec(min_train_rows=30))

    def test_nonchronological_input_fails_closed(self) -> None:
        frame = self._frame().copy()
        frame.index = list(range(len(frame) - 1, -1, -1))
        with self.assertRaises(ValueError):
            evaluate_short_challengers(frame, ["f1", "f2"], ShortOOSSpec(min_train_rows=30))


if __name__ == "__main__":
    unittest.main()
