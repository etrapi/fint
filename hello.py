def main():
    import numpy as np
    import pandas as pd
    import polars as pl
    import time
    from fint.indicators import indicators as ind
    from fint.indicators import config

    # Random seed
    np.random.seed(42)
    prices_1d = np.random.rand(1000).astype(np.float64)
    prices_2d = np.random.rand(1000, 2000).astype(np.float64)
    prices_pd_series = pd.Series(prices_1d)
    prices_pl_series = pl.Series(prices_1d)
    dates = pd.date_range(start="2000-01-01", periods=1000, freq="D")
    prices_pd_1d = pd.DataFrame(prices_1d, index=dates)
    prices_pd_2d = pd.DataFrame(prices_2d, index=dates)
    prices_pl_1d = pl.DataFrame(prices_pd_1d.reset_index())
    prices_pl_2d = pl.DataFrame(prices_pd_2d.reset_index())
    

    # NUMPY AND NUMBA VERSIONS CAN HANDLE 2D ARRAY WHILE TA-LIB CAN ONLY HANDLE 1D ARRAY
    # NUMPY 1D ARRAY
    if False:
        print("\nNUMPY 1D ARRAY")
        print("Prices:", prices_1d[:10])
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        print("NumPy SMA:", ind.SMA(prices_1d, 3)[:10])
        print("NumPy ROCOHLC:", ind.ROCOHLC(prices_1d, prices_1d, prices_1d, prices_1d, 2)[:10])
        print("NumPy time:", time.time() - t)

        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        print("Numba SMA:", ind.SMA(prices_1d, 3)[:10])
        print("Numba ROCOHLC:", ind.ROCOHLC(prices_1d, prices_1d, prices_1d, prices_1d, 2)[:10])
        print("Numba time:", time.time() - t)
        
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        print("TA-Lib SMA:", ind.SMA(prices_1d, 3)[:10])
        print("TA-Lib time:", time.time() - t)


    # NUMPY 2D ARRAY
    if False:
        print("\nNUMPY 2D ARRAY")
        print("Prices:", prices_2d[:10])
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        #print("NumPy SMA:", ind.SMA(prices, 3)[:10])
        #print("NumPy ROCOHLC:", ind.ROCOHLC(prices, prices, prices, prices, 2)[:10])
        #print("NumPy VOLUME_ACCU:", ind.VOLUME_ACCU(prices, prices)[:10])
        #print("NumPy OPEN_PCT_CHANGE:", ind.OPEN_PCT_CHANGE(prices, prices, prices)[:10])
        #print("NumPy YESTERDAY_CLOSE_PCT_CHANGE:", ind.YESTERDAY_CLOSE_PCT_CHANGE(prices, prices)[:10])
        print("NumPy TIME_FROM_START:", ind.TIME_FROM_START(prices_2d, prices_2d)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        #print("Numba SMA:", ind.SMA(prices, 3)[:10])
        #print("Numba ROCOHLC:", ind.ROCOHLC(prices, prices, prices, prices, 2)[:10])
        #print("Numba VOLUME_ACCU:", ind.VOLUME_ACCU(prices, prices)[:10])
        #print("Numba OPEN_PCT_CHANGE:", ind.OPEN_PCT_CHANGE(prices, prices, prices)[:10])
        #print("Numba YESTERDAY_CLOSE_PCT_CHANGE:", ind.YESTERDAY_CLOSE_PCT_CHANGE(prices, prices)[:10])
        print("Numba TIME_FROM_START:", ind.TIME_FROM_START(prices_2d, prices_2d)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        # loop prices array on axis 1
        for i in range(prices_2d.shape[1]):
            ind.SMA(prices_2d[:, i], 3)
        print("TA-Lib SMA time:", time.time() - t)
    
    # PANDAS SERIES
    if False:
        print("\nPANDAS: SERIES")
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        print("NumPy SMA:", ind.SMA(prices_pd_series, 3)[:10])
        print("NumPy ROCOHLC:", ind.ROCOHLC(prices_pd_series, prices_pd_series, prices_pd_series, prices_pd_series, 2)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        print(type(ind.SMA(prices_pd_series, 3)))
        print("Numba SMA:", ind.SMA(prices_pd_series, 3)[:10])
        print("Numba ROCOHLC:", ind.ROCOHLC(prices_pd_series, prices_pd_series, prices_pd_series, prices_pd_series, 2)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        print("TA-Lib SMA:", ind.SMA(prices_pd_series, 3)[:10])
        print("TA-Lib time:", time.time() - t)

    # PANDAS DATAFRAME 1D
    if False:
        print("\nPANDAS: DATAFRAME 1D")
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        print("NumPy SMA:", ind.SMA(prices_pd_1d, 3)[:10])
        #print("NumPy ROCOHLC:", ind.ROCOHLC(prices_pd, prices_pd, prices_pd, prices_pd, 2)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        print("Numba SMA:", ind.SMA(prices_pd_1d, 3)[:10])
        #print("Numba ROCOHLC:", ind.ROCOHLC(prices_pd, prices_pd, prices_pd, prices_pd, 2)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        print("TA-Lib SMA:", ind.SMA(prices_pd_1d, 3)[:10])
        print("TA-Lib time:", time.time() - t)

    # PANDAS DATAFRAME 2D
    if True:
        print("\nPANDAS: DATAFRAME 2D")
        print("Prices:\n", prices_pd_2d)
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        print("NumPy SMA:", ind.SMA(prices_pd_2d, 3)[:10])
        #print("NumPy ROCOHLC:", ind.ROCOHLC(prices_pd, prices_pd, prices_pd, prices_pd, 2)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        print("Numba SMA:", ind.SMA(prices_pd_2d, 3)[:10])
        #print("Numba ROCOHLC:", ind.ROCOHLC(prices_pd, prices_pd, prices_pd, prices_pd, 2)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        # loop prices array on axis 1
        for i in range(prices_pd_2d.shape[1]):
            ind.SMA(prices_pd_2d.iloc[:, i], 3)
        print("TA-Lib SMA time:", time.time() - t)
    
    # POLARS SERIES
    if False:
        print("\nPOLARS: SERIES")
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        print("NumPy SMA:", ind.SMA(prices_pl_series, 3)[:10])
        print("NumPy ROCOHLC:", ind.ROCOHLC(prices_pl_series, prices_pl_series, prices_pl_series, prices_pl_series, 2)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        print("Numba SMA:", ind.SMA(prices_pl_series, 3)[:10])
        print("Numba ROCOHLC:", ind.ROCOHLC(prices_pl_series, prices_pl_series, prices_pl_series, prices_pl_series, 2)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        print("TA-Lib SMA:", ind.SMA(prices_pl_series, 3)[:10])
        print("TA-Lib time:", time.time() - t)

    # POLARS DATAFRAME 1D
    if False:
        print("\nPOLARS: DATAFRAME 1D")
        print("Prices:\n", prices_pl_1d)
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        print("NumPy SMA:", ind.SMA(prices_pl_1d, 3)[:10])
        print("NumPy ROCOHLC:", ind.ROCOHLC(prices_pl_1d, prices_pl_1d, prices_pl_1d, prices_pl_1d, 2)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        print("Numba SMA:", ind.SMA(prices_pl_1d, 3)[:10])
        print("Numba ROCOHLC:", ind.ROCOHLC(prices_pl_1d, prices_pl_1d, prices_pl_1d, prices_pl_1d, 2)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        print("TA-Lib SMA:", ind.SMA(prices_pl_1d, 3)[:10])
        print("TA-Lib time:", time.time() - t)

    # POLARS DATAFRAME 2D
    if False:
        print("\nPOLARS: DATAFRAME 2D")
        print("Prices:\n", prices_pl_2d)
        # 1. NumPy
        t = time.time()
        config.ACTIVE_BACKEND = "numpy"
        print("NumPy SMA:", ind.SMA(prices_pl_2d, 3)[:10])
        print("NumPy ROCOHLC:", ind.ROCOHLC(prices_pl_2d, prices_pl_2d, prices_pl_2d, prices_pl_2d, 2)[:10])
        print("NumPy time:", time.time() - t)
        # 2. Numba
        t = time.time()
        config.ACTIVE_BACKEND = "numba"
        print("Numba SMA:", ind.SMA(prices_pl_2d, 3)[:10])
        print("Numba ROCOHLC:", ind.ROCOHLC(prices_pl_2d, prices_pl_2d, prices_pl_2d, prices_pl_2d, 2)[:10])
        print("Numba time:", time.time() - t)
        # 3. TA-Lib
        t = time.time()
        config.ACTIVE_BACKEND = "talib"
        # loop prices array on axis 1
        for i in range(prices_pl_2d.shape[1]):
            ind.SMA(prices_pl_2d[:, i], 3)
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
