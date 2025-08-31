import numpy as np
import pytest
import talib
from fint.indicators.backends import talib_impl


def test_sma_talib_basic():
    """Test simple moving average with TA-Lib backend."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    expected = talib.SMA(data, timeperiod=3)
    result = talib_impl.sma_talib(data, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


def test_sma_talib_short_input():
    """Test SMA with input shorter than period."""
    data = np.array([1.0, 2.0], dtype=np.float64)
    result = talib_impl.sma_talib(data, 3)
    assert np.all(np.isnan(result))


def test_rsi_talib_basic():
    """Test RSI calculation with TA-Lib backend."""
    # Simple increasing sequence
    data = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
    expected = talib.RSI(data, timeperiod=2)
    result = talib_impl.rsi_talib(data, 2)
    np.testing.assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


def test_rsi_talib_edge_cases():
    """Test RSI edge cases with TA-Lib."""
    # All equal values
    data = np.ones(10, dtype=np.float64)
    result = talib_impl.rsi_talib(data, 2)
    # RSI should be 50 when there are no price changes
    assert np.all(result[~np.isnan(result)] == 50.0)
