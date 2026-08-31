from __future__ import annotations

import pandas as pd
import pytest

from research.deep_mnq_source_contract import (
    contract_receipt,
    normalize_deep_bar_timestamps,
)


def _norm(*values: str) -> list[pd.Timestamp]:
    out = normalize_deep_bar_timestamps(pd.Series(list(values)))
    return list(out)


def test_winter_new_york_close_labels_normalize_to_utc_bar_start() -> None:
    assert _norm("2026-01-21 00:01:00", "2026-01-21 00:02:00") == [
        pd.Timestamp("2026-01-21T05:00:00Z"),
        pd.Timestamp("2026-01-21T05:01:00Z"),
    ]


def test_summer_dst_close_labels_normalize_to_utc_bar_start() -> None:
    assert _norm("2025-06-10 00:01:00", "2025-06-10 00:02:00") == [
        pd.Timestamp("2025-06-10T04:00:00Z"),
        pd.Timestamp("2025-06-10T04:01:00Z"),
    ]


def test_contract_receipt_records_dst_aware_bar_label_semantics() -> None:
    receipt = contract_receipt()
    assert receipt["source_timestamp_timezone"] == "America/New_York"
    assert receipt["source_bar_label"] == "right_close"
    assert receipt["normalized_timestamp"] == "UTC bar start"


def test_timezone_aware_input_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="already timezone-aware"):
        normalize_deep_bar_timestamps(pd.Series(pd.to_datetime(["2026-01-21T05:00:00Z"])))
