import pandas as pd
import numpy as np


if __name__ == '__main__':
    import db_operations as dbop
    from constants import *
    import data_process as dp
    import feature_engineering as FE

    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20130101
    df = dbop.create_df(cursor, STOCK_DAY[TABLE],
                        start=start,
                        where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600352.SH'",
                        )
    df = dp.proc_stock_d(dp.prepare_stock_d(df))
    print(df.shape)
    import collect
    # pool = list(collect.get_stock_pool())[:10]
    # print(len(pool))
    # print(pool)
    # df = df.loc[FE.IDX[:,pool],:]
    print(df.shape)

    pd.set_option("display.max_columns",10)

    # expected = pd.concat([move(n, group) for _, group in df[cols].groupby(level="code")])\
    #     .dropna().sort_index()

    delta = 1e-8
    # cols = ["close","vol","amt"]
    # cols = ["open", "high", "low", "close","vol","amt"]
    k = 5

    targets = [{"period": 20, "func": "max", "col": "high"},
               {"period": 20, "func": "min", "col": "low"},
               {"period": 20, "func": "avg", "col": ""},
               {"period": 5, "func": "max", "col": "high"},
               {"period": 5, "func": "min", "col": "low"},
               {"period": 5, "func": "avg", "col": ""},
               ]

    import time
    t0 = time.time()
    # actual = FE.k_line_batch(k,df[cols],sort=False).dropna().sort_index()
    actual,_ = FE.stock_d_FE_batch(df,targets=targets)
    actual = actual.sort_index().fillna(0)
    print(time.time()-t0)
    t0=time.time()
    expected = pd.concat([FE.stock_d_FE(group,targets=targets)[0] for _,group in df.groupby(level="code")])\
        .sort_index().fillna(0)
    print(time.time()-t0)
    print(expected.shape,actual.shape)
    print(sorted(actual.columns))
    print("\n")
    print(sorted(expected.columns))
    print(sorted(actual.columns)==sorted(expected.columns))

    # print((expected == actual).all().all())
    print(((expected - actual).abs() < delta).all().all())
    # print((expected - actual).abs())
    # print(((expected - actual).abs() >=delta).sum().sum())
    # row_mask = ~((expected - actual).abs() < delta).all(axis=1)
    # column_mask = ~((expected - actual).abs() < delta).all(axis=0)
    # print(row_mask.sum(),column_mask.sum())
    # print(actual.loc[row_mask,column_mask])
    # print(expected.loc[row_mask,column_mask])