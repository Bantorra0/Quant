import pandas as pd
import numpy as np


def fe_test():
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20130101
    df = dbop.create_df(cursor, STOCK_DAY[TABLE],
                        start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600352.SH'",
                        )
    df = dp.proc_stock_d(dp.prepare_stock_d(df))
    print(df.shape)
    pool = sorted(collect.get_stock_pool()["code"])[:5]
    print(len(pool))
    print(pool)
    df = df.loc[FE.IDX[:, pool], :]
    print(df.shape)

    pd.set_option("display.max_columns", 10)

    delta = 5e-4
    epsilon = 1e-4
    # cols = ["close","vol","amt"]
    # cols = ["open", "high", "low", "close","vol","amt"]

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
    # actual,_ = FE.stock_d_FE_batch(df,targets=targets)
    k = 20
    actual, _ = FE.mp_batch(df=df, target=FE.stock_d_FE_batch, batch_size=k, targets=targets)
    print(time.time() - t0)
    print(actual.info(memory_usage="deep"))
    actual = actual.sort_index().replace({float("inf"): 9999, np.nan: 0})
    t0 = time.time()
    expected = pd.concat([FE.stock_d_FE(group, targets=targets)[0] for _, group in df.groupby(level="code")])
    print(time.time() - t0)
    expected = expected.sort_index().replace({float("inf"): 9999, np.nan: 0})
    print(expected.shape, actual.shape)
    print(sorted(actual.columns))
    print("\n")
    print(sorted(expected.columns))
    print(sorted(actual.columns) == sorted(expected.columns))

    error = (expected - actual).abs() / (expected + epsilon)
    test_cond = error < delta

    # print((expected == actual).all().all())
    print(test_cond.all().all())
    # print((expected - actual).abs())
    print((~test_cond).sum().sum())
    column_mask = ~test_cond.all(axis=0)
    row_mask = ~test_cond.all(axis=1)
    print(row_mask.sum(), column_mask.sum())
    print(actual.loc[row_mask, column_mask])
    print(expected.loc[row_mask, column_mask])
    print(error.loc[row_mask, column_mask])


if __name__ == '__main__':
    import db_operations as dbop
    from constants import *
    import data_process as dp
    import feature_engineering as FE
    import collect


    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20190101
    df = dbop.create_df(cursor, STOCK_DAY[TABLE],
                        start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600352.SH'",
                        )
    df = dp.proc_stock_d(dp.prepare_stock_d(df))
    print(df.shape)
    # pool = sorted(collect.get_stock_pool()["code"])[:100]
    # print(len(pool))
    # print(pool)
    # df = df.loc[FE.IDX[pool,:],:]
    # print(df.shape)

    pd.set_option("display.max_columns",10)

    # cols = ["close","vol","amt"]
    # cols = ["open", "high", "low", "close","vol","amt"]

    targets = [{"period": 20, "func": "max", "col": "high"},
               {"period": 20, "func": "min", "col": "low"},
               {"period": 20, "func": "avg", "col": ""},
               {"period": 5, "func": "max", "col": "high"},
               {"period": 5, "func": "min", "col": "low"},
               {"period": 5, "func": "avg", "col": ""},
               ]


    import time
    t0 = time.time()
    fe_list={
        "not_in_X":False,
        "basic":False,
        "kma":False,
        "k_line":False,
        "rolling":True,
    }
    k = 235
    actual,_ = FE.mp_batch(df=df,target=FE.stock_d_FE_batch,batch_size=k,
                           num_reserved_cpu=1,targets=targets,fe_list=fe_list)
    print(time.time()-t0)
    print(actual.info(memory_usage="deep"))
