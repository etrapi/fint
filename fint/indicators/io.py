
from typing import Any, Callable

import pandas as pd
import numpy as np
try:
    import numba
except ImportError:
    numba = None

try:
    import polars as pl
except ImportError:
    pl = None
from functools import wraps


def io_wrapper(func):
    """Wrap a function to convert Pandas/Polars inputs to numpy, operate in 2D, and reconstruct outputs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import polars as pl

        new_args = []
        orig_info = []
        is_polars_any = False

        # --- Convert inputs to numpy 2D ---
        for a in args:
            info = {"orig_type": type(a), "was_1d": False}

            if isinstance(a, pd.DataFrame):
                numeric_cols = a.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found in the input Pandas DataFrame")
                arr = a[numeric_cols].to_numpy(dtype=np.float64)
                info.update({
                    "type": "pd_df",
                    "columns": numeric_cols,
                    "index": a.index
                })
                new_args.append(arr)

            elif isinstance(a, pd.Series):
                arr = a.to_numpy(dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                    info["was_1d"] = True
                info.update({"type": "pd_series", "index": a.index, "name": a.name})
                new_args.append(arr)

            elif isinstance(a, pl.DataFrame):
                is_polars_any = True
                numeric_cols = [col for col, dtype in zip(a.columns, a.dtypes) 
                                if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)]
                if not numeric_cols:
                    raise ValueError("No numeric columns found in the input Polars DataFrame")
                arr = a[numeric_cols].to_numpy(dtype=np.float64)
                info.update({
                    "type": "pl_df",
                    "columns": numeric_cols,
                    "orig_cols": a.columns
                })
                new_args.append(arr)

            elif isinstance(a, pl.Series):
                is_polars_any = True
                arr = a.to_numpy(dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                    info["was_1d"] = True
                info.update({"type": "pl_series", "name": a.name})
                new_args.append(arr)

            elif hasattr(a, "__array__"):
                arr = np.asarray(a, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                    info["was_1d"] = True
                info["type"] = "array"
                new_args.append(arr)

            else:
                info["type"] = "other"
                new_args.append(a)

            orig_info.append(info)

        # --- Call the original function ---
        result = func(*new_args, **kwargs)

        # --- Reconstruct output ---
        if not args:
            return result

        first_info = orig_info[0]
        first_arg = args[0]

        # Handle 1D -> 2D conversion revert
        if first_info.get("was_1d", False):
            result = result.ravel()

        # Polars reconstruction
        if is_polars_any:
            import polars as pl
            if first_info["type"] == "pl_df":
                cols = first_info.get("columns", [])
                out_cols = {col: result[:, i] for i, col in enumerate(cols)} if result.ndim > 1 else {cols[0]: result}
                # Keep original non-numeric columns
                for c in first_info.get("orig_cols", []):
                    if c not in cols:
                        out_cols[c] = first_arg[c]
                return pl.DataFrame(out_cols)
            elif first_info["type"] == "pl_series":
                return pl.Series(result, name=first_info.get("name", "result"))

        # Pandas reconstruction
        if isinstance(first_arg, pd.DataFrame):
            cols = first_info.get("columns", [])
            return pd.DataFrame(result, index=first_info.get("index", first_arg.index), columns=cols)
        elif isinstance(first_arg, pd.Series):
            return pd.Series(result, index=first_info.get("index", first_arg.index), name=first_info.get("name", "result"))

        # Numpy or other fallback
        return result

    return wrapper


def optional_numba(func):
    """Apply Numba JIT if Numba is available."""
    if numba is None:
        import warnings
        warnings.warn("Numba not found. Falling back to regular numpy calculations.")
        return func
    try:
        return numba.njit(func)
    except Exception as e:
        import warnings
        warnings.warn(f"Numba JIT compilation failed: {e}. Falling back to regular Python.")
        return func


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return x

def ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, 0]
    return x