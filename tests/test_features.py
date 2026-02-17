"""Unit tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_features,
)


def _make_sample_df(n: int = 35) -> pd.DataFrame:
    """Create a small synthetic DataFrame mimicking M5 structure."""
    return pd.DataFrame(
        {
            "day_num": range(1, n + 1),
            "sales": np.arange(1, n + 1, dtype=float),
            "wday": [((i % 7) + 1) for i in range(n)],  # cycles 1-7
            "month": [1] * n,
            "year": [2016] * n,
            "event_name_1": [None] * (n - 3) + ["SuperBowl", None, "Easter"],
            "item_id": ["ITEM_1"] * n,
            "store_id": ["CA_1"] * n,
            "cal_date": pd.date_range("2016-01-01", periods=n),
        }
    )


class TestLagFeatures:
    def test_lag_values_correct(self):
        """Lag values should match a manual shift."""
        df = _make_sample_df()
        df = add_lag_features(df, lags=[7])

        # Row at index 7 (day_num=8) should have lag_7 = sales at index 0 = 1.0
        assert df.iloc[7]["sales_lag_7"] == 1.0
        # Row at index 10 (day_num=11) should have lag_7 = sales at index 3 = 4.0
        assert df.iloc[10]["sales_lag_7"] == 4.0

    def test_lag_columns_created(self):
        """Each lag value produces a named column."""
        df = _make_sample_df()
        df = add_lag_features(df, lags=[7, 14, 28])
        assert "sales_lag_7" in df.columns
        assert "sales_lag_14" in df.columns
        assert "sales_lag_28" in df.columns

    def test_lag_produces_nans(self):
        """First `lag` rows should be NaN."""
        df = _make_sample_df()
        df = add_lag_features(df, lags=[7])
        assert df["sales_lag_7"].isna().sum() == 7


class TestRollingFeatures:
    def test_rolling_features_no_leakage(self):
        """shift(1) must prevent the current row from entering its own rolling window."""
        df = _make_sample_df(n=15)
        df = add_rolling_features(df, windows=[3])

        # Row i uses shifted sales [0..i-1], rolling(3) over the last 3 of those.
        # At index 4 (day_num=5): shifted = [NaN,1,2,3,4], rolling(3) over [2,3,4] = mean 3.0
        assert df.iloc[4]["sales_rolling_mean_3"] == pytest.approx(3.0)
        # Current row's sales=5 should NOT appear in the mean
        assert df.iloc[4]["sales_rolling_mean_3"] != pytest.approx((3.0 + 4.0 + 5.0) / 3)

    def test_rolling_columns_created(self):
        df = _make_sample_df()
        df = add_rolling_features(df, windows=[7, 28])
        for w in [7, 28]:
            assert f"sales_rolling_mean_{w}" in df.columns
            assert f"sales_rolling_std_{w}" in df.columns


class TestCalendarFeatures:
    def test_is_weekend(self):
        """wday 1 (Sat) and 2 (Sun) should be weekend."""
        df = pd.DataFrame(
            {
                "wday": [1, 2, 3, 4, 5, 6, 7],
                "event_name_1": [None] * 7,
            }
        )
        df = add_calendar_features(df)
        assert df["is_weekend"].tolist() == [1, 1, 0, 0, 0, 0, 0]

    def test_has_event(self):
        """Non-null event_name_1 should flag as 1."""
        df = pd.DataFrame(
            {
                "wday": [3, 3, 3],
                "event_name_1": [None, "SuperBowl", None],
            }
        )
        df = add_calendar_features(df)
        assert df["has_event"].tolist() == [0, 1, 0]


class TestBuildFeatures:
    def test_build_features_columns(self):
        """build_features should produce all expected feature columns."""
        df = _make_sample_df()
        df = build_features(df)

        expected = [
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_28",
            "sales_rolling_mean_7",
            "sales_rolling_mean_28",
            "sales_rolling_std_7",
            "sales_rolling_std_28",
            "is_weekend",
            "has_event",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_build_features_preserves_existing_columns(self):
        """Original columns should still be present after build_features."""
        df = _make_sample_df()
        original_cols = set(df.columns)
        df = build_features(df)
        assert original_cols.issubset(set(df.columns))
