import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants as const
import db_operations as dbop
import data_prepare as dp
import ml_model
import customized_obj as cus_obj

import xgboost.sklearn as xgb
import lightgbm.sklearn as lgbm
import sklearn as sk

import datetime
import time
import multiprocessing as mp



if __name__ == '__main__':
    db_type = "sqlite3"

    # Feature engineering and save datasets.
    targets = [{"period": 20, "fun": "max", "col": "high"},
               {"period": 20, "fun": "min", "col": "low"},
               {"period": 5, "fun": "max", "col": "high"},
               {"period": 5, "fun": "min", "col": "low"},
               # {"period": 20, "fun": "mean", "col": ""}
               ]

    time_delta = datetime.timedelta(days=1)
    test_start = "2018-01-01"
    train_length = 3000
    max_feature_length = 1000

    cursor = dbop.connect_db(db_type=db_type).cursor()
    num_files = 2

    df_stock_basic = dbop.create_df(cursor, const.STOCK_BASIC[const.TABLE])
    h_stock_pool = sorted(df_stock_basic[df_stock_basic["is_hs"] == "H"]["code"])
    s_stock_pool = sorted(df_stock_basic[df_stock_basic["is_hs"] == "S"]["code"])
    print(len(h_stock_pool),len(s_stock_pool))

    train_bound = datetime.datetime.strptime(test_start, const.DATE_FORMAT) - train_length * time_delta
    train_bound = datetime.datetime.strftime(train_bound, const.DATE_FORMAT)

    lower_bound = datetime.datetime.strptime(train_bound, const.DATE_FORMAT) - max_feature_length * time_delta
    lower_bound = datetime.datetime.strftime(lower_bound, const.DATE_FORMAT)
    print(test_start,train_bound,lower_bound)

    t0 = time.time()
    # num_p = mp.cpu_count()
    # p_pool = mp.Pool(processes=mp.cpu_count())
    df_feature, df_not_in_X, cols_category, enc = ml_model.gen_data(
        targets=targets,
        lowerbound=lower_bound,
        start=train_bound,
        stock_pool=h_stock_pool)

    print("df_all:", df_feature.shape)
    trading_date_idxes = df_feature.index.unique().sort_values(ascending=True)

    # X = ml_model.gen_X(df_feature, df_not_in_X.columns)
    X = df_feature

    paras = [("y_l_rise", {"pred_period": 20, "is_high": True, "is_clf": False,"threshold":0.2}, df_not_in_X),
             ("y_l_decline", {"pred_period": 20, "is_high": False, "is_clf": False, "threshold":0.2}, df_not_in_X),
             ("y_s_rise", {"pred_period": 5, "is_high": True, "is_clf": False,"threshold":0.1}, df_not_in_X),
             ("y_s_decline", {"pred_period": 5, "is_high": False, "is_clf": False,"threshold":0.1}, df_not_in_X),
             ]

    Y = pd.concat([ml_model.gen_y(v2, **v1) for _, v1, v2 in paras], axis=1)
    Y.columns = [k for k, _, _ in paras]
    Y.index = X.index
    Y["y_l"] = Y.apply(
        lambda r:r["y_l_rise"] if r["y_l_rise"]> -r["y_l_decline"] else r["y_l_decline"],
        axis=1)
    print(X.shape, Y.shape, Y.columns)

    # with open(r"datasets/hgt_X.csv","w") as f:
    #     X.to_csv(f)
    # with open(r"datasets/hgt_Y.csv","w") as f:
    #     Y.to_csv(f)
    # with open(r"datasets/hgt_other_info.csv", "w") as f:
    #     df_not_in_X.to_csv(f)

    # X.to_hdf(r"datasets/hgt_X.hdf",key="X")

    X.to_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")
    Y.to_parquet(r"datasets/hgt_Y.parquet",engine="fastparquet")
    df_not_in_X.to_parquet(r"datasets/hgt_other_info.parquet",engine="fastparquet")

    print(time.time()-t0)
    print(X.info(memory_usage='deep'))
    print("float64:",list(X.columns[X.dtypes=="float64"]))
    print("int64:",list(X.columns[X.dtypes == "int64"]))
    print("object:",list(X.columns[X.dtypes == "object"]))

    # Test if future data is used.
    X["code"] = df_not_in_X["code"]
    X_latest_day = X.loc[trading_date_idxes[-1]]
    print(sorted(X_latest_day.columns[X_latest_day.isnull().any(axis=0)]))
    print(X_latest_day.shape)
    for k,v in X_latest_day.isnull().sum().sort_index().iteritems():
        if v>0:
            print(k,v)
    pd.set_option("display.max_columns",10)
    print(X_latest_day[X_latest_day["(open/p40mv_10k_open-1)"].isnull()][[
        "code","open","close","(open/p60max_open-1)",
        "(open/p40mv_10k_open-1)"]])

    del X["code"]
    print(X.info(memory_usage='deep'))
    print(Y.dtypes)