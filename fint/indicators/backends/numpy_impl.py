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


# Registro dinámico
register_indicator("SMA", "numpy", io_wrapper(sma_numpy))
register_indicator("RSI", "numpy", io_wrapper(rsi_numpy))
register_indicator("VWAP", "numpy", io_wrapper(vwap_numpy))
register_indicator("ROCOHLC", "numpy", io_wrapper(roc_ohlc_numpy))