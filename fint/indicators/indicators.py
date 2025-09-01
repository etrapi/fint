from .registry import get_indicator

# Importar e inicializar backends
from .backends import numpy_impl
from .backends import numba_impl
from .backends import talib_impl   # << nuevo

# API pública
def SMA(x, period, backend=None):
    return get_indicator("SMA", backend)(x, period)

def RSI(x, period=14, backend=None):
    return get_indicator("RSI", backend)(x, period)

def ROC(x, period=14, backend=None):
    return get_indicator("ROC", backend)(x, period)

def VWAP(x, period=14, backend=None):
    return get_indicator("VWAP", backend)(x, period)

def ROCOHLC(open, high, low, close, length, backend=None):
    return get_indicator("ROCOHLC", backend)(open, high, low, close, length)

def VOLUME_ACCU(volume, start_time_mask, backend=None):
    return get_indicator("VOLUME_ACCU", backend)(volume, start_time_mask)

def OPEN_PCT_CHANGE(open, close, start_time_mask, backend=None):
    return get_indicator("OPEN_PCT_CHANGE", backend)(open, close, start_time_mask)

def YESTERDAY_CLOSE_PCT_CHANGE(close, start_time_mask, backend=None):
    return get_indicator("YESTERDAY_CLOSE_PCT_CHANGE", backend)(close, start_time_mask)

def TIME_FROM_START(open, start_time_mask, backend=None):
    return get_indicator("TIME_FROM_START", backend)(open, start_time_mask)   
