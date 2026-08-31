from __future__ import annotations

import unittest

import pandas as pd

from research.mnq_short_opportunity_targets import (
    ShortOpportunitySpec,
    assert_short_feature_fence,
    materialize_short_opportunity_targets,
)


class ShortOpportunityTargetsTest(unittest.TestCase):
    def test_downside_economics_and_excursions_are_short_specific(self) -> None:
        frame = pd.DataFrame({
            "high": [101, 100, 99, 98, 98],
            "low": [99, 97, 95, 94, 93],
            "close": [100, 98, 96, 95, 94],
        })
        result = materialize_short_opportunity_targets(
            frame, ShortOpportunitySpec(horizon_bars=3, cost_points=1.0, min_edge_points=1.0)
        )
        self.assertEqual(result.loc[0, "short_label"], 1)
        self.assertEqual(result.loc[0, "short_forward_points"], 5.0)
        self.assertEqual(result.loc[0, "short_mfe_points"], 6.0)
        self.assertEqual(result.loc[0, "short_mae_points"], 0.0)
        self.assertEqual(result.loc[0, "target_resolution_row"], 3)

    def test_adverse_excursion_can_reject_terminally_profitable_short(self) -> None:
        frame = pd.DataFrame({
            "high": [101, 106, 101, 99],
            "low": [99, 98, 95, 94],
            "close": [100, 99, 97, 95],
        })
        result = materialize_short_opportunity_targets(
            frame,
            ShortOpportunitySpec(horizon_bars=3, cost_points=1.0, adverse_limit_points=4.0),
        )
        self.assertEqual(result.loc[0, "short_forward_points"], 5.0)
        self.assertEqual(result.loc[0, "short_mae_points"], 6.0)
        self.assertEqual(result.loc[0, "short_label"], 0)

    def test_unresolved_tail_is_not_labeled(self) -> None:
        frame = pd.DataFrame({"high": [2, 2, 2], "low": [1, 1, 1], "close": [1.5, 1.4, 1.3]})
        result = materialize_short_opportunity_targets(frame, ShortOpportunitySpec(horizon_bars=2))
        self.assertTrue(pd.isna(result.loc[1, "short_label"]))

    def test_future_outcomes_are_feature_fenced(self) -> None:
        assert_short_feature_fence(["ret_1", "rv_120"])
        with self.assertRaises(ValueError):
            assert_short_feature_fence(["ret_1", "short_mfe_points"])


if __name__ == "__main__":
    unittest.main()
