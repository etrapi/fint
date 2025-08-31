import numpy as np
from ..registry import register_indicator
from ..io import io_wrapper, optional_numba
from ..backends.numpy_impl import roc_ohlc_numpy

def sma_numba(x: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[:] = np.nan
    cumsum = 0.0
    for i in range(len(x)):
        cumsum += x[i]
        if i >= period:
            cumsum -= x[i - period]
        if i >= period - 1:
            out[i] = cumsum / period
    return out

def rsi_numba(x: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    gains = np.empty(n-1, dtype=np.float64)
    losses = np.empty(n-1, dtype=np.float64)

    for i in range(1, n):
        delta = x[i] - x[i-1]
        gains[i-1] = delta if delta > 0 else 0.0
        losses[i-1] = -delta if delta < 0 else 0.0

    for i in range(period, n):
        avg_gain = np.mean(gains[i-period:i])
        avg_loss = np.mean(losses[i-period:i])
        rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
        out[i] = 100 - (100 / (1 + rs))

    return out

def roc_ohlc_numba(
    open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int
):
    ''' Equivalente a roc_ohlc_numpy '''
    return roc_ohlc_numpy(open, high, low, close, length)

# Registro dinámico
register_indicator("SMA", "numba", io_wrapper(optional_numba(sma_numba)))
register_indicator("RSI", "numba", io_wrapper(optional_numba(rsi_numba)))
register_indicator("ROCOHLC", "numba", io_wrapper(optional_numba(roc_ohlc_numba)))