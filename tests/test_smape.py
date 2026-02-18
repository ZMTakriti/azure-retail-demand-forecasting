"""Unit tests for MAE and RMSE metric calculations."""

import numpy as np
import pytest


def _mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class TestMAE:
    def test_perfect_prediction(self):
        assert _mae([100, 200, 300], [100, 200, 300]) == pytest.approx(0.0)

    def test_known_value(self):
        # |100-80| = 20 → MAE = 20
        assert _mae([100], [80]) == pytest.approx(20.0)

    def test_multiple_values(self):
        # errors: 10, 10, 10 → MAE = 10
        assert _mae([100, 200, 150], [110, 210, 160]) == pytest.approx(10.0)

    def test_handles_zeros(self):
        assert _mae([0, 100], [0, 100]) == pytest.approx(0.0)

    def test_numpy_arrays(self):
        result = _mae(np.array([100, 200]), np.array([110, 190]))
        assert isinstance(result, float)
        assert result > 0

    def test_empty_arrays(self):
        assert np.isnan(_mae([], []))


class TestRMSE:
    def test_perfect_prediction(self):
        assert _rmse([100, 200, 300], [100, 200, 300]) == pytest.approx(0.0)

    def test_known_value(self):
        # error=20 → RMSE = 20
        assert _rmse([100], [80]) == pytest.approx(20.0)

    def test_penalises_large_errors(self):
        # RMSE > MAE when errors vary
        y_true = [100, 100]
        y_pred = [90, 50]  # errors: 10, 50
        assert _rmse(y_true, y_pred) > _mae(y_true, y_pred)

    def test_numpy_arrays(self):
        result = _rmse(np.array([100, 200]), np.array([110, 190]))
        assert isinstance(result, float)
        assert result > 0
