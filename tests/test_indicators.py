import numpy as np
import pandas as pd
import pytest
from fint.indicators.indicators import SMA, RSI
from fint.indicators import config


def test_sma_numpy_backend(sample_prices):
    """Test SMA with numpy backend."""
    config.ACTIVE_BACKEND = "numpy"
    result = SMA(sample_prices, 3)
    assert len(result) == len(sample_prices)
    assert np.isnan(result[0])  # First values should be NaN
    assert not np.isnan(result[-1])  # Last value should be calculated


def test_sma_numba_backend(sample_prices):
    """Test SMA with numba backend."""
    config.ACTIVE_BACKEND = "numba"
    result = SMA(sample_prices, 3)
    assert len(result) == len(sample_prices)
    assert np.isnan(result[0])
    assert not np.isnan(result[-1])


def test_sma_talib_backend(sample_prices):
    """Test SMA with TA-Lib backend."""
    config.ACTIVE_BACKEND = "talib"
    result = SMA(sample_prices, 3)
    assert len(result) == len(sample_prices)
    assert np.isnan(result[0])
    assert not np.isnan(result[-1])


def test_sma_pandas_input(sample_pandas_series):
    """Test SMA with pandas Series input."""
    result = SMA(sample_pandas_series, 3)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_pandas_series)
    assert result.name == sample_pandas_series.name


def test_rsi_basic(sample_prices):
    """Test RSI calculation."""
    result = RSI(sample_prices, 3)
    assert len(result) == len(sample_prices)
    # First few values should be NaN due to warmup period
    assert np.isnan(result[0])
    # RSI should be between 0 and 100
    assert 0 <= result[-1] <= 100 or np.isnan(result[-1])


def test_backend_override(sample_prices):
    """Test backend override in function call."""
    numpy_result = SMA(sample_prices, 3, backend="numpy")
    numba_result = SMA(sample_prices, 3, backend="numba")
    talib_result = SMA(sample_prices, 3, backend="talib")
    
    # Results should be similar but might have small numerical differences
    np.testing.assert_allclose(
        numpy_result[~np.isnan(numpy_result)],
        numba_result[~np.isnan(numba_result)],
        rtol=1e-7
    )
    
    # Compare with TA-Lib if available
    if not np.all(np.isnan(talib_result)):
        np.testing.assert_allclose(
            numpy_result[~np.isnan(numpy_result)],
            talib_result[~np.isnan(talib_result)],
            rtol=1e-7
        )
