import numpy as np
from ..registry import register_indicator
from ..io import io_wrapper, optional_numba
from ..backends.numpy_impl import (
    roc_ohlc_numpy,
    volume_accu_indicator_numpy,
    open_pct_change_numpy,
    yesterday_close_pct_change_numpy,
    time_from_start_mask_numpy
)

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

def volume_accu_indicator_numba(volume: np.ndarray, start_time_mask: np.ndarray):
    ''' Equivalente a volume_accu_indicator_numpy '''
    return volume_accu_indicator_numpy(volume, start_time_mask)

def open_pct_change_numba(
    open: np.ndarray, close: np.ndarray, start_time_mask: np.ndarray
):
    ''' Equivalente a open_pct_change_numpy '''
    return open_pct_change_numpy(open, close, start_time_mask)

def yesterday_close_pct_change_numba(
    close: np.ndarray, start_time_mask: np.ndarray
):
    ''' Equivalente a yesterday_close_pct_change_numpy '''
    return yesterday_close_pct_change_numpy(close, start_time_mask)

def time_from_start_mask_numba(open: np.ndarray, start_time_mask: np.ndarray):
    ''' Equivalente a time_from_start_mask_numpy '''
    return time_from_start_mask_numpy(open, start_time_mask)

# Registro dinámico
register_indicator("SMA", "numba", io_wrapper(optional_numba(sma_numba)))
register_indicator("RSI", "numba", io_wrapper(optional_numba(rsi_numba)))
register_indicator("ROCOHLC", "numba", io_wrapper(optional_numba(roc_ohlc_numba)))
register_indicator("VOLUME_ACCU", "numba", io_wrapper(optional_numba(volume_accu_indicator_numba)))
register_indicator("OPEN_PCT_CHANGE", "numba", io_wrapper(optional_numba(open_pct_change_numba)))
register_indicator("YESTERDAY_CLOSE_PCT_CHANGE", "numba", io_wrapper(optional_numba(yesterday_close_pct_change_numba)))
register_indicator("TIME_FROM_START", "numba", io_wrapper(optional_numba(time_from_start_mask_numba)))
