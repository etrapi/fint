
from typing import Any, Callable

import pandas as pd
import numpy as np
try:
    import numba
except ImportError:
    numba = None
from functools import wraps

def io_wrapper(func):
    """Convert input Pandas/Polars to numpy and back after calculation.
    
    Args:
        func (Callable): Function to wrap.
    
    Returns:
        Callable: Wrapped function.
    """
    @wraps(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        import polars as pl
        
        new_args = []
        index = None
        is_polars = False
        
        for a in args:
            if isinstance(a, pd.DataFrame):
                numeric_cols = a.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found in the input DataFrame")
                new_args.append(a[numeric_cols[0]].to_numpy(dtype=np.float64))
                index = a.index
                
            elif isinstance(a, pd.Series):
                new_args.append(a.to_numpy(dtype=np.float64))
                index = a.index
                
            elif isinstance(a, (pl.DataFrame, pl.Series)):
                is_polars = True
                if isinstance(a, pl.DataFrame):
                    # Get first numeric column
                    numeric_cols = [col for col, dtype in zip(a.columns, a.dtypes) 
                                  if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)]
                    if not numeric_cols:
                        raise ValueError("No numeric columns found in the input Polars DataFrame")
                    values = a[numeric_cols[0]].to_numpy().astype(np.float64)
                    index = a[a.columns[0]].to_numpy()  # Use first column as index
                    new_args.append(values)
                else:  # pl.Series
                    new_args.append(a.to_numpy(dtype=np.float64))
                    index = a.to_numpy()  # Use series values as index
                    
            elif hasattr(a, '__array__'):
                new_args.append(np.asarray(a, dtype=np.float64))
            else:
                new_args.append(a)
                
        result = func(*new_args, **kwargs)
        
        if not args:
            return result
            
        first = args[0]
        
        if is_polars:
            if isinstance(first, pl.DataFrame):
                # Create new DataFrame with original first column as index and result as values
                return pl.DataFrame({
                    first.columns[0]: index,
                    "result": result
                })
            return pl.Series(result, name="result")
            
        if isinstance(first, (pd.Series, pd.DataFrame)):
            return type(first)(result, index=index if index is not None else first.index)
            
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