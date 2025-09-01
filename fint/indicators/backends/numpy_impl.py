import numpy as np
from ..registry import register_indicator
from ..io import io_wrapper

def sma_numpy(x: np.ndarray, period: int) -> np.ndarray:
    if len(x) < period:
        return np.full_like(x, np.nan)
    cumsum = np.cumsum(x, dtype=float)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    return np.concatenate([
        np.full(period-1, np.nan),
        cumsum[period-1:] / period
    ])

def rsi_numpy(x: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(x)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.convolve(gain, np.ones(period)/period, mode="valid")
    avg_loss = np.convolve(loss, np.ones(period)/period, mode="valid")

    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))

    return np.concatenate([
        np.full(period, np.nan),
        rsi
    ])

def vwap_numpy(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    cum_pv = np.cumsum(prices * volumes)
    cum_vol = np.cumsum(volumes)
    return cum_pv / cum_vol

def roc_ohlc_numpy(
    open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int
):
# Input validation
    if not (open.shape == high.shape == low.shape == close.shape):
        raise ValueError("All input arrays must have the same shape")
    if length <= 0 or length >= len(close):
        raise ValueError("Length must be positive and less than array length")
    
    ohlc4 = (open + high + low + close) / 4
    actual = ohlc4[length:, :]
    previous = ohlc4[:-length, :]
    roc = np.divide(actual - previous, previous, 
                where=previous != 0, 
                out=np.full_like(actual, np.nan))
    indicator = np.full_like(close, np.nan)
    indicator[length:, :] = roc
    return indicator

def volume_accu_indicator_numpy(volume: np.ndarray, start_time_mask: np.ndarray):
    volume_acu = np.full_like(volume, np.nan)
    for column_index in range(volume.shape[1]):
        for row_index in range(volume.shape[0]):
            if start_time_mask[row_index,column_index]:# added column_¡ndex from original
                volume_acu[row_index, column_index] = volume[row_index, column_index]
            else:
                volume_acu[row_index, column_index] = (
                    volume_acu[row_index - 1, column_index]
                    + volume[row_index, column_index]
                )
    return volume_acu

def open_pct_change_numpy(
    open: np.ndarray, close: np.ndarray, start_time_mask: np.ndarray
):
    open_pct_change = np.full_like(close, np.nan)
    open_market_row = 0
    for column_index in range(open.shape[1]):
        for row_index in range(open.shape[0]):
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

# yesterday rth close pct change
def yesterday_close_pct_change_numpy(
    close: np.ndarray, start_time_mask: np.ndarray
): 
    close_pct_change = np.full_like(close, np.nan)
    close_market_row = 0
    for column_index in range(close.shape[1]):
        for row_index in range(close.shape[0]):
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


def time_from_start_mask_numpy(open: np.ndarray, start_time_mask: np.ndarray):
    time_from_start = np.full_like(open, np.nan)
    for column_index in range(open.shape[1]):
        for row_index in range(open.shape[0]):
            if start_time_mask[row_index, column_index]: # added column_¡ndex from original
                time_from_start[row_index, column_index] = 0
            else:
                time_from_start[row_index, column_index] = time_from_start[row_index - 1, column_index] + 1
    return time_from_start



# Registro dinámico
register_indicator("SMA", "numpy", io_wrapper(sma_numpy))
register_indicator("RSI", "numpy", io_wrapper(rsi_numpy))
register_indicator("VWAP", "numpy", io_wrapper(vwap_numpy))
register_indicator("ROCOHLC", "numpy", io_wrapper(roc_ohlc_numpy))
register_indicator("VOLUME_ACCU", "numpy", io_wrapper(volume_accu_indicator_numpy))
register_indicator("OPEN_PCT_CHANGE", "numpy", io_wrapper(open_pct_change_numpy))
register_indicator("YESTERDAY_CLOSE_PCT_CHANGE", "numpy", io_wrapper(yesterday_close_pct_change_numpy))
register_indicator("TIME_FROM_START", "numpy", io_wrapper(time_from_start_mask_numpy))
