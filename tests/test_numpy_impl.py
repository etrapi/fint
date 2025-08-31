import numpy as np
import pytest
from fint.indicators.backends import numpy_impl


def test_sma_numpy_basic():
    """Test simple moving average with numpy backend."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
    result = numpy_impl.sma_numpy(data, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


def test_sma_numpy_short_input():
    """Test SMA with input shorter than period."""
    data = np.array([1.0, 2.0])
    result = numpy_impl.sma_numpy(data, 3)
    assert np.all(np.isnan(result))


def test_rsi_numpy_basic():
    """Test RSI calculation with numpy backend."""
    # Simple increasing sequence
    data = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    result = numpy_impl.rsi_numpy(data, 2)
    # First 3 should be nan (period=2 + 1 for first diff)
    assert np.all(np.isnan(result[:3]))
    # RSI should be 100 for constantly increasing sequence
    assert result[-1] == 100.0


def test_vwap_numpy():
    """Test volume weighted average price calculation."""
    prices = np.array([10.0, 11.0, 12.0])
    volumes = np.array([100, 200, 300])
    expected = np.array([10.0, (10*100 + 11*200)/300, (10*100 + 11*200 + 12*300)/600])
    result = numpy_impl.vwap_numpy(prices, volumes)
    np.testing.assert_allclose(result, expected, rtol=1e-9)
