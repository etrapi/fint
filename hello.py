def main():
    import numpy as np
    import pandas as pd
    import time
    from fint.indicators import indicators as ind
    from fint.indicators import config

    # Random seed
    np.random.seed(42)

    # NUMPY 1D ARRAY
    if False:
        prices = np.random.rand(10000000).astype(np.float64)
        print("Prices:", prices[:10])
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        #print("NumPy SMA:", ind.SMA(prices, 3)[:10])
        print("NumPy ROCOHLC:", ind.ROCOHLC(prices, prices, prices, prices, 2)[:10])
        print("NumPy SMA time:", time.time() - t)

        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        #print("Numba SMA:", ind.SMA(prices, 3)[:10])
        print("Numba ROCOHLC:", ind.ROCOHLC(prices, prices, prices, prices, 2)[:10])
        print("Numba SMA time:", time.time() - t)
        
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        print("TA-Lib SMA:", ind.SMA(prices, 3)[:10])
        print("TA-Lib SMA time:", time.time() - t)


    # NUMPY 2D ARRAY
    if True:
        prices = np.random.rand(100000, 2000).astype(np.float64)
        print("Prices:", prices[:10])
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        #print("NumPy SMA:", ind.SMA(prices, 3)[:10])
        #print("NumPy ROCOHLC:", ind.ROCOHLC(prices, prices, prices, prices, 2)[:10])
        #print("NumPy VOLUME_ACCU:", ind.VOLUME_ACCU(prices, prices)[:10])
        #print("NumPy OPEN_PCT_CHANGE:", ind.OPEN_PCT_CHANGE(prices, prices, prices)[:10])
        #print("NumPy YESTERDAY_CLOSE_PCT_CHANGE:", ind.YESTERDAY_CLOSE_PCT_CHANGE(prices, prices)[:10])
        print("NumPy TIME_FROM_START:", ind.TIME_FROM_START(prices, prices)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        #print("Numba SMA:", ind.SMA(prices, 3)[:10])
        #print("Numba ROCOHLC:", ind.ROCOHLC(prices, prices, prices, prices, 2)[:10])
        #print("Numba VOLUME_ACCU:", ind.VOLUME_ACCU(prices, prices)[:10])
        #print("Numba OPEN_PCT_CHANGE:", ind.OPEN_PCT_CHANGE(prices, prices, prices)[:10])
        #print("Numba YESTERDAY_CLOSE_PCT_CHANGE:", ind.YESTERDAY_CLOSE_PCT_CHANGE(prices, prices)[:10])
        print("Numba TIME_FROM_START:", ind.TIME_FROM_START(prices, prices)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        # loop prices array on axis 1
        for i in range(prices.shape[1]):
            ind.SMA(prices[:, i], 3)
        print("TA-Lib SMA time:", time.time() - t)






    # # 4. Override por función
    # print("Mix: NumPy RSI:", ind.RSI(prices, 3, backend="numpy")[:10])
    # print("Mix: TA-Lib RSI:", ind.RSI(prices, 3, backend="talib")[:10])

    # dates = pd.date_range(start="2021-01-01", periods=100, freq="D")
    # data_np = np.random.rand(100).astype(np.float64)
    # data_pd = pd.DataFrame({ "open": data_np, "close": data_np}, index=dates)
    # data_pl = pl.DataFrame({"date": dates, "open": data_np, "close": data_np})
    # print("Pandas:\n", data_pd)
    # print("Polars:\n", data_pl)
    
    # print("Numpy SMA:\n", ind.SMA(data_np, 3)[:10])
    # print("Pandas SMA:\n", ind.SMA(data_pd, 3)[:10])
    # print("Polars SMA:\n", ind.SMA(data_pl, 3)[:10])
if __name__ == "__main__":
    main()
