"""Unit tests for SMAPE metric."""

import numpy as np
import pytest

from src.model import smape


class TestSmape:
    """Tests for the SMAPE function."""

    def test_perfect_prediction(self):
        """SMAPE should be 0 for perfect predictions."""
        y_true = [100, 200, 300]
        y_pred = [100, 200, 300]
        assert smape(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        """SMAPE should be symmetric: smape(a, b) == smape(b, a)."""
        y_true = [100, 200]
        y_pred = [120, 180]
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))

    def test_known_value(self):
        """Test against a known SMAPE calculation."""
        # For y_true=100, y_pred=80:
        # SMAPE = 2 * |80-100| / (|100| + |80|) = 40/180 = 0.222...
        # As percentage: 22.22%
        y_true = [100]
        y_pred = [80]
        expected = 100 * 2 * 20 / 180  # ~22.22%
        assert smape(y_true, y_pred) == pytest.approx(expected, rel=1e-3)

    def test_multiple_values(self):
        """Test SMAPE with multiple values."""
        y_true = [100, 200, 150]
        y_pred = [110, 190, 160]
        # Manual calculation:
        # |110-100|=10, denom=210, contrib=20/210
        # |190-200|=10, denom=390, contrib=20/390
        # |160-150|=10, denom=310, contrib=20/310
        # SMAPE = 100/3 * sum
        result = smape(y_true, y_pred)
        assert 0 < result < 20  # Sanity check - should be small

    def test_handles_zeros(self):
        """SMAPE should handle zero values without division error."""
        y_true = [0, 100]
        y_pred = [10, 100]
        # Should not raise, epsilon prevents division by zero
        result = smape(y_true, y_pred)
        assert result > 0

    def test_upper_bound(self):
        """SMAPE should not exceed 200%."""
        y_true = [100, 200]
        y_pred = [0, 0]
        result = smape(y_true, y_pred)
        assert result <= 200

    def test_numpy_arrays(self):
        """Should work with numpy arrays."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        result = smape(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0

    def test_empty_arrays(self):
        """Should handle empty arrays gracefully."""
        # Empty arrays return 0 (no error to measure)
        result = smape([], [])
        assert result == 0.0
