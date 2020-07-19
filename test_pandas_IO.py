import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants as const
import db_operations as dbop
import data_prepare as dp
import io_operations
import ml_model
import customized_obj as cus_obj

import xgboost.sklearn as xgb
import lightgbm.sklearn as lgbm
import sklearn.preprocessing as preproc
import sklearn.metrics as metrics
import sklearn as sk

import datetime
import time
import multiprocessing as mp
import pickle

if __name__ == '__main__':
    db_type = "sqlite3"

    targets = [{"period": 20, "fun": "max", "col": "high"},
               {"period": 20, "fun": "min", "col": "low"},
               {"period": 5, "fun": "max", "col": "high"},
               {"period": 5, "fun": "min", "col": "low"},
               # {"period": 20, "fun": "mean", "col": ""}
               ]

    # time_delta = datetime.timedelta(days=1)
    # test_start = "2018-01-01"
    # train_length = 3000
    # max_feature_length = 1000

    cursor = dbop.connect_db(db_type=db_type).cursor()

    df_stock_basic = dbop.create_df(cursor, const.STOCK_BASIC[const.TABLE])
    h_stock_pool = sorted(
        df_stock_basic[df_stock_basic["is_hs"] == "H"]["code"])
    s_stock_pool = sorted(
        df_stock_basic[df_stock_basic["is_hs"] == "S"]["code"])
    print(len(h_stock_pool), len(s_stock_pool))

    # train_bound = datetime.datetime.strptime(test_start, const.DATE_FORMAT) - train_length * time_delta
    # train_bound = datetime.datetime.strftime(train_bound, const.DATE_FORMAT)
    # lower_bound = datetime.datetime.strptime(train_bound, const.DATE_FORMAT) - max_feature_length * time_delta
    # lower_bound = datetime.datetime.strftime(lower_bound, const.DATE_FORMAT)
    # print(test_start,train_bound,lower_bound)

    lower_bound = "2013-01-01"
    start = "2018-01-01"
    end = "2018-07-01"
    upper_bound = "2019-01-01"

    t0 = time.time()
    # num_p = mp.cpu_count()
    # p_pool = mp.Pool(processes=mp.cpu_count())
    df_feature, df_not_in_X, cols_category, enc = ml_model.gen_data(
        targets=targets, lowerbound=lower_bound, start=start, end=end,
        upperbound=upper_bound, stock_pool=h_stock_pool)

    print("df_all:", df_feature.shape)
    trading_date_idxes = df_feature.index.unique().sort_values(ascending=True)

    # X = ml_model.gen_X(df_feature, df_not_in_X.columns)
    X = df_feature

    paras = [("y_l_rise", {"pred_period": 20, "is_high": True, "is_clf": False,
                           "threshold": 0.2}, df_not_in_X), ("y_l_decline", {
        "pred_period": 20, "is_high": False, "is_clf": False,
        "threshold": 0.2}, df_not_in_X), ("y_s_rise",
                                          {"pred_period": 5, "is_high": True,
                                           "is_clf": False, "threshold": 0.1},
                                          df_not_in_X), ("y_s_decline",
                                                         {"pred_period": 5,
                                                          "is_high": False,
                                                          "is_clf": False,
                                                          "threshold": 0.1},
                                                         df_not_in_X), ]

    # paras = [("y_l", {"pred_period": 20, "is_high": True, "is_clf": False,
    #                        "threshold":0.1}, df_all)]
    Y = pd.concat([ml_model.gen_y(v2, **v1) for k, v1, v2 in paras], axis=1)
    Y.columns = [k for k, _, _ in paras]
    Y.index = X.index
    Y["y_l"] = Y.apply(
        lambda r: r["y_l_rise"] if r["y_l_rise"] > -r["y_l_decline"] else r[
            "y_l_decline"], axis=1)
    print(X.shape, Y.shape, Y.columns)

    print(X.info(memory_usage='deep'))
    print("float64:", list(X.columns[X.dtypes == "float64"]))
    print("int64:", list(X.columns[X.dtypes == "int64"]))
    print("object:", list(X.columns[X.dtypes == "object"]))

    f_path = r"datasets/stock_d.hdf"
    f_path = io_operations._add_suffix_to_file_names(f_path,
                                                     datetime.datetime.now().strftime("%Y-%m-%d"))
    year = start[:4]
    start_M_D = start[5:].replace("-", "")
    end_M_D = end[5:].replace("-", "")
    key = "{0}/{1}-{2}".format(year, start_M_D, end_M_D)
    print(key)
    print(f_path)

    X.to_hdf(f_path, key="X/" + key)
    Y.to_hdf(f_path, key="Y/" + key)
    df_not_in_X.to_hdf(f_path, key="other/" + key)

    store = pd.HDFStore(f_path)
    print(store.keys())
    print(store["X/" + key].shape)
    print(store["Y/" + key].shape)
    print(store["other/" + key].shape)

    # X.to_hdf(r"datasets/hgt_X.hdf",key="X")

    # X.to_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")  # Y.to_parquet(r"datasets/hgt_Y.parquet",engine="fastparquet")  # df_not_in_X.to_parquet(r"datasets/hgt_other.parquet",engine="fastparquet")

    # files = {0:r"datasets/hgt.hdf"}  # suffix = datetime.datetime.now().strftime('%Y-%m-%d')  # f_hdf_name = ml_model.add_suffix_to_file_names(files, suffix)[0]  #  # t0=time.time()  # X.to_hdf(f_hdf_name, key="X")  # print("Write X in hdf:",time.time()-t0)  #  # t0 = time.time()  # Y.to_hdf(f_hdf_name, key="Y")  # print("Write Y in hdf:",time.time()-t0)  #  # t0 = time.time()  # df_not_in_X.to_hdf(f_hdf_name, key="other")  # print("Write other in hdf:",time.time()-t0)  #  #  # files = {  #     "Y": r"datasets/hgt_Y.parquet",  #     "X": r"datasets/hgt_X.parquet",  #     "other": r"datasets/hgt_other.parquet"  # }  #  # files = ml_model.add_suffix_to_file_names(files, suffix)  #  # t0 = time.time()  # X.to_parquet(files["X"], engine="fastparquet")  # print("Write X in parquet with fastparquet",time.time()-t0)  #  # t0=time.time()  # Y.to_parquet(files["Y"], engine="fastparquet")  # print("Write Y in parquet with fastparquet", time.time() - t0)  #  # t0=time.time()  # df_not_in_X.to_parquet(files["other"], engine="fastparquet")  # print("Write other in parquet with fastparquet", time.time() - t0)  #  # dataset_info = {"feature_names": list(X.columns), "target_names": list(Y.columns),  #                 "other_names": list(df_not_in_X.columns)}  #  # print(time.time()-t0)  #  # # X["code"] = df_not_in_X["code"]  # # X_latest_day = X.loc[trading_date_idxes[-1]]  # # print(sorted(X_latest_day.columns[X_latest_day.isnull().any(axis=0)]))  # # print(X_latest_day.shape)  # # for k,v in X_latest_day.isnull().sum().sort_index().iteritems():  # #     if v>0:  # #         print(k,v)  # # pd.set_option("display.max_columns",10)  # # print(X_latest_day[X_latest_day["(open/p40mv_10k_open-1)"].isnull()][[  # #     "code","open","close","(open/p60max_open-1)",  # #     "(open/p40mv_10k_open-1)"]])  # # del X["code"]  #  # print(X.info(memory_usage='deep'))  # print(Y.dtypes)  #  # N = len(Y.index)  # row_i_idxes = np.arange(N)  # for i in range(4):  #     np.random.shuffle(row_i_idxes)  #  # dataset_info["shuffle"]= row_i_idxes  # f_pickle_name = r"datasets/hgt_dataset_info"  # f_pickle_name = ml_model.add_suffix_to_file_names({0:f_pickle_name},suffix)[0]  # with open(f_pickle_name, mode="wb") as f:  #     t0 = time.time()  #     pickle.dump(dataset_info, f)  #     print("Pickle dump time:",time.time()-t0)  #  # t0 = time.time()  # X=pd.read_hdf(f_hdf_name, "X")  # print("Read X in hdf5:",time.time()-t0)  # print(X.shape)  # t0 = time.time()  # Y=pd.read_hdf(f_hdf_name, "Y")  # print("Read Y in hdf5:",time.time()-t0)  # print(Y.shape)  # t0 = time.time()  # df_other_info = pd.read_hdf(f_hdf_name, "other")  # print("Read other info in hdf5:", time.time() - t0)  # print(df_other_info.shape)  #  # t0 = time.time()  # X = pd.read_parquet(files["X"])  # print("Read X in parquet:", time.time() - t0)  # print(X.shape)  # t0 = time.time()  # Y = pd.read_parquet(files["Y"])  # print("Read Y in parquet:", time.time() - t0)  # print(Y.shape)  # t0 = time.time()  # df_other_info = pd.read_parquet(files["other"])  # print("Read other info in parquet:", time.time() - t0)  # print(df_other_info.shape)  #  # t0 = time.time()  # X = pd.read_parquet(files["X"],engine="fastparquet")  # print("Read X in parquet,engine=fastparquet:", time.time() - t0)  # print(X.shape,X.info(memory_usage="deep"))  # t0 = time.time()  # Y = pd.read_parquet(files["Y"],engine="fastparquet")  # print("Read Y in parquet,engine=fastparquet:", time.time() - t0)  # print(Y.shape)  # t0 = time.time()  # df_other_info = pd.read_parquet(files["other"],engine="fastparquet")  # print("Read other info in parquet,engine=fastparquet:", time.time() - t0)  # print(df_other_info.shape)

    # files ={  #     "Y":r"datasets/hgt_Y.parquet",  #     "X":r"datasets/hgt_X.parquet",  #     "other":r"datasets/hgt_other.parquet"  # }  #  # with open(r"datasets/hgt_dataset_info", mode="rb") as f:  #     dataset_info = pickle.load(f)  # row_i_idxes = dataset_info["shuffle"]  # N = len(row_i_idxes)  # k_split = 100  # print(row_i_idxes)  # subsample_idxes = row_i_idxes[:(N // k_split)]  #  # t_start_read_Y = time.time()  # # X = pd.read_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")  # Y = pd.read_parquet(files["Y"], engine="fastparquet")  # print("Reading parquet file {0} in {1:.2f}".format(files["Y"],time.time() - t_start_read_Y))  # print(Y.shape, Y.columns)  # # print(Y.iloc[:5])  # Y_subsample = Y.iloc[subsample_idxes]  # del Y  #  # t_start_read_X = time.time()  # X = pd.read_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")  # print("Reading parquet file {0} in {1:.2f}".format(files["X"],time.time() - t_start_read_X))  # X_subsample = X.iloc[subsample_idxes]  # del X  #  # t_start_read_other = time.time()  # df_other = pd.read_parquet(r"datasets/hgt_other.parquet",engine="fastparquet")  # print("Reading parquet file {0} in {1:.2f}".format(files["other"],time.time() - t_start_read_other))  # df_other_subsample = df_other.iloc[subsample_idxes]  # del df_other  #  # test_start = "2018-08-01"  # trading_dates = Y_subsample.index.unique().sort_values(ascending=True)  # train_dates = trading_dates[trading_dates<test_start][:-21]  # test_dates = trading_dates[trading_dates>=test_start]  #  # X_train = X_subsample.loc[train_dates]  # Y_train = Y_subsample.loc[train_dates]  # X_test = X_subsample.loc[test_dates]  # Y_test = Y_subsample.loc[test_dates]  #  # ycol = "y_l"  # cond = Y_test[ycol].notnull()  # X_test = X_test[cond]  # Y_test = Y_test[cond]  # print(X_train.shape,X_test.shape)  #  # reg = lgbm.LGBMRegressor(n_estimators=50,num_leaves=31,max_depth=12,  #                          min_child_samples=30,random_state=0)  #  # train_start = time.time()  # cols_category = ["area", "industry", "market", "exchange", "is_hs"]  # reg.fit(X_train,Y_train[ycol],categorical_feature=cols_category)  # print("Train time:", time.time() - train_start)  # print(reg.score(X_test,Y_test[ycol]))  # df_feature_importance = ml_model.get_feature_importance(reg,X_test.columns)  # pd.set_option("display.max_columns",10)  # pd.set_option("display.max_rows",256)  # print(df_feature_importance[df_feature_importance[  #     "importance_raw"]>0].round({  #     "importance_percent":2}))  #  # ycol2 = "y_l_rise"  # ml_model.pred_interval_summary(reg, X_test, Y_test[ycol])  # ml_model.pred_interval_summary(reg, X_test, Y_test[ycol2])  # plt.show()