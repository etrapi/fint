import numpy as np
import pytest
from fint.indicators.backends import numba_impl


def test_sma_numba_basic():
    """Test simple moving average with numba backend."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0], dtype=np.float64)
    result = numba_impl.sma_numba(data, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


def test_sma_numba_short_input():
    """Test SMA with input shorter than period."""
    data = np.array([1.0, 2.0], dtype=np.float64)
    result = numba_impl.sma_numba(data, 3)
    assert np.all(np.isnan(result))


def test_rsi_numba_basic():
    """Test RSI calculation with numba backend."""
    # Simple increasing sequence
    data = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
    result = numba_impl.rsi_numba(data, 2)
    # First 3 should be nan (period=2 + 1 for first diff)
    assert np.all(np.isnan(result[:3]))
    # RSI should be 100 for constantly increasing sequence
    assert np.isclose(result[-1], 100.0)


def test_rsi_numba_all_losses():
    """Test RSI with all decreasing prices."""
    data = np.array([10.0, 9.0, 8.0, 7.0, 6.0], dtype=np.float64)
    result = numba_impl.rsi_numba(data, 2)
    assert np.all(np.isnan(result[:3]))
    # RSI should be 0 for constantly decreasing sequence
    assert np.isclose(result[-1], 0.0)
