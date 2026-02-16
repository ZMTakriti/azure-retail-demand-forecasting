"""
Model training utilities for M5 demand forecasting.
"""

import numpy as np


def smape(y_true, y_pred) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    SMAPE is bounded between 0% and 200%, making it suitable for
    comparing forecast accuracy across different series scales.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        SMAPE value as a percentage (0-200).

    Example
    -------
    >>> smape([100, 200], [110, 180])
    14.285714285714285
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return 0.0

    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return 100.0 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / denominator)
