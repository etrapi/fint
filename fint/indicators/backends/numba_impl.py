import numpy as np
from numba import njit, prange
from ..registry import register_indicator
from ..io import io_wrapper, optional_numba

@njit(cache=True, fastmath=True, parallel=True)
def sma_numba(x, period):
    n_rows, n_cols = x.shape
    out = np.empty_like(x)
    out[:] = np.nan
    
    for col in prange(n_cols):
        if n_rows < period:
            continue
        cumsum = np.empty(n_rows, dtype=np.float64)
        cumsum[0] = x[0, col]
        for i in range(1, n_rows):
            cumsum[i] = cumsum[i-1] + x[i, col]
        
        for i in range(period-1, n_rows):
            if i == period-1:
                out[i, col] = cumsum[i] / period
            else:
                out[i, col] = (cumsum[i] - cumsum[i-period]) / period
    return out

@njit(cache=True, parallel=True, fastmath=True)
def roc_ohlc_numba(open, high, low, close, length):
    # Input validation
    if not (open.shape == high.shape == low.shape == close.shape):
        raise ValueError("All input arrays must have the same shape")
    # if shape is not 
    if length <= 0 or length >= len(close):
        raise ValueError("Length must be positive and less than array length")

    n_rows, n_cols = open.shape
    indicator = np.full_like(close, np.nan)
    
    ohlc4 = (open + high + low + close) / 4
    
    for col in prange(n_cols):
        for i in range(length, n_rows):
            prev = ohlc4[i - length, col]
            if prev != 0.0:
                indicator[i, col] = (ohlc4[i, col] - prev) / prev
    return indicator

def volume_accu_numba(volume, start_time_mask):
    n_rows, n_cols = volume.shape
    volume_acu = np.empty_like(volume, dtype=np.float64)
    
    for col in prange(n_cols):
        for row in range(n_rows):
            if row == 0 or start_time_mask[row, col]:
                volume_acu[row, col] = volume[row, col]
            else:
                volume_acu[row, col] = volume_acu[row - 1, col] + volume[row, col]
    
    return volume_acu

@njit(cache=True, parallel=True, fastmath=True)
def open_pct_change_numba(
    open: np.ndarray, close: np.ndarray, start_time_mask: np.ndarray
):
    open_pct_change = np.full_like(close, np.nan)
    open_market_row = 0
    for column_index in prange(open.shape[1]):
        for row_index in prange(open.shape[0]):
            if start_time_mask[row_index, column_index]:# added column_¡ndex from original
                open_pct_change[row_index, column_index] = 0
                open_market_row = row_index
            else:
                close_now = close[row_index, column_index]
                open_market_start = open[open_market_row, column_index]
                open_pct_change[row_index, column_index] = (
                    (close_now - open_market_start) / open_market_start
                )
        open_market_row = 0
    return open_pct_change

@njit(cache=True, parallel=True, fastmath=True)
def yesterday_close_pct_change_numba(
    close: np.ndarray, start_time_mask: np.ndarray
): 
    close_pct_change = np.full_like(close, np.nan)
    close_market_row = 0
    for column_index in prange(close.shape[1]):
        for row_index in prange(close.shape[0]):
            if start_time_mask[row_index, column_index]:# added column_¡ndex from original
                close_pct_change[row_index, column_index] = 0
                close_market_row = row_index
            else:
                close_last= close[-1, column_index]
                close_yesterday = close[close_market_row, column_index]
                close_pct_change[row_index, column_index] = (
                    (close_last - close_yesterday) / close_yesterday
                )
    return close_pct_change

@njit(cache=True, parallel=True, fastmath=True)
def time_from_start_mask_numba(open: np.ndarray, start_time_mask: np.ndarray):
    time_from_start = np.full_like(open, np.nan)
    for column_index in prange(open.shape[1]):
        for row_index in prange(open.shape[0]):
            if start_time_mask[row_index, column_index]: # added column_¡ndex from original
                time_from_start[row_index, column_index] = 0
            else:
                time_from_start[row_index, column_index] = time_from_start[row_index - 1, column_index] + 1
    return time_from_start

# Registro dinámico
register_indicator("SMA", "numba", io_wrapper(sma_numba))
register_indicator("ROCOHLC", "numba", io_wrapper(roc_ohlc_numba))
register_indicator("VOLUME_ACCU", "numba", io_wrapper(volume_accu_numba))
register_indicator("OPEN_PCT_CHANGE", "numba", io_wrapper(open_pct_change_numba))
register_indicator("YESTERDAY_CLOSE_PCT_CHANGE", "numba", io_wrapper(yesterday_close_pct_change_numba))
register_indicator("TIME_FROM_START", "numba", io_wrapper(time_from_start_mask_numba))
