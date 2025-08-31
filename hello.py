def main():
    import numpy as np
    import pandas as pd
    import polars as pl
    from fint.indicators.indicators import SMA, RSI
    from fint.indicators import config

    prices = np.random.rand(1000).astype(np.float64)
    print("Prices:", prices[:10])

    # 1. NumPy
    config.ACTIVE_BACKEND = "numpy"
    print("NumPy SMA:", SMA(prices, 3)[:10])

    # 2. Numba
    config.ACTIVE_BACKEND = "numba"
    print("Numba SMA:", SMA(prices, 3)[:10])

    # 3. TA-Lib
    config.ACTIVE_BACKEND = "talib"
    print("TA-Lib SMA:", SMA(prices, 3)[:10])

    # 4. Override por función
    print("Mix: NumPy RSI:", RSI(prices, 3, backend="numpy")[:10])
    print("Mix: TA-Lib RSI:", RSI(prices, 3, backend="talib")[:10])

    dates = pd.date_range(start="2021-01-01", periods=100, freq="D")
    data_np = np.random.rand(100).astype(np.float64)
    data_pd = pd.DataFrame({ "open": data_np, "close": data_np}, index=dates)
    data_pl = pl.DataFrame({"date": dates, "open": data_np, "close": data_np})
    print("Pandas:\n", data_pd)
    print("Polars:\n", data_pl)
    
    print("Numpy SMA:\n", SMA(data_np, 3)[:10])
    print("Pandas SMA:\n", SMA(data_pd, 3)[:10])
    print("Polars SMA:\n", SMA(data_pl, 3)[:10])
if __name__ == "__main__":
    main()
