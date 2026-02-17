"""Feature engineering for M5 demand forecasting.

Pure pandas functions that operate on a single-item DataFrame
sorted by day_num. All functions expect a 'sales' column.
"""

import pandas as pd


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged sales columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'sales' column sorted by day_num.
    lags : list[int], optional
        Lag periods to create. Defaults to [7, 14, 28].

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new ``sales_lag_{n}`` columns.
    """
    if lags is None:
        lags = [7, 14, 28]
    for n in lags:
        df[f"sales_lag_{n}"] = df["sales"].shift(n)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling mean and std of sales.

    Uses ``shift(1)`` before rolling to prevent data leakage — the
    current row's sales value is never included in its own features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'sales' column sorted by day_num.
    windows : list[int], optional
        Rolling window sizes. Defaults to [7, 28].

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new rolling columns.
    """
    if windows is None:
        windows = [7, 28]
    shifted = df["sales"].shift(1)
    for w in windows:
        df[f"sales_rolling_mean_{w}"] = shifted.rolling(w).mean()
        df[f"sales_rolling_std_{w}"] = shifted.rolling(w).std()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'wday' (M5 convention: 1=Sat, 2=Sun) and
        'event_name_1' columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``is_weekend`` and ``has_event`` columns.
    """
    df["is_weekend"] = (df["wday"].isin([1, 2])).astype(int)
    df["has_event"] = df["event_name_1"].notna().astype(int)
    return df


def build_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Orchestrate all feature engineering steps.

    Calls :func:`add_lag_features`, :func:`add_rolling_features`, and
    :func:`add_calendar_features` in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Single-item DataFrame sorted by day_num with columns
        ``sales``, ``wday``, and ``event_name_1``.
    lags : list[int], optional
        Passed to :func:`add_lag_features`.
    windows : list[int], optional
        Passed to :func:`add_rolling_features`.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all engineered features.
    """
    if lags is None:
        lags = [7, 14, 28]
    if windows is None:
        windows = [7, 28]
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=windows)
    df = add_calendar_features(df)
    return df
