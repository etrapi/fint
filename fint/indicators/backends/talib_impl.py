import numpy as np
import talib
from ..registry import register_indicator
from ..io import io_wrapper

@io_wrapper
def sma_talib(x: np.ndarray, period: int) -> np.ndarray:
    return talib.SMA(x, timeperiod=period)

@io_wrapper
def rsi_talib(x: np.ndarray, period: int = 14) -> np.ndarray:
    return talib.RSI(x, timeperiod=period)

# Registro dinámico
register_indicator("SMA", "talib", sma_talib)
register_indicator("RSI", "talib", rsi_talib)