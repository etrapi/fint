from typing import Callable, Dict
from .config import ACTIVE_BACKEND

# Estructura: {"SMA": {"numpy": func_numpy, "numba": func_numba, ...}}
_registry: Dict[str, Dict[str, Callable]] = {}

def register_indicator(name: str, backend: str, func: Callable):
    """Registra un indicador para un backend."""
    if name not in _registry:
        _registry[name] = {}
    _registry[name][backend] = func

def get_indicator(name: str, backend: str = None) -> Callable:
    """Obtiene la implementación de un indicador según el backend."""
    backend = backend or ACTIVE_BACKEND
    try:
        return _registry[name][backend]
    except KeyError:
        raise ValueError(f"No implementation for {name} in backend '{backend}'")
