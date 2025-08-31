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
