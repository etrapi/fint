import numpy as np
import talib
from ..registry import register_indicator
from ..io import io_wrapper, ensure_1d


def sma_talib(x: np.ndarray, period: int) -> np.ndarray:
    x = ensure_1d(x)
    return talib.SMA(x, timeperiod=period)


def rsi_talib(x: np.ndarray, period: int = 14) -> np.ndarray:
    x = ensure_1d(x)
    return talib.RSI(x, timeperiod=period)

# Registro dinámico
register_indicator("SMA", "talib", io_wrapper(sma_talib))
register_indicator("RSI", "talib", io_wrapper(rsi_talib))