import datetime
import time
import os
import pickle
import schedule
import gc

import lightgbm.sklearn as lgbm
import pandas as pd

import customized_obj as cus_obj
import io_operations as IO_op
import ml_model as ml
from collect import *
import db_operations as dbop
from constants import *
import data_process as dp
from feature_engineering import return_script


def collect_data():
    db_type = "sqlite3"

    update_stock_basic()

    # index_pool = dbop.get_all_indexes()
    index_pool = get_index_pool()
    print(index_pool)
    # init_table(const.INDEX_DAY[const.TABLE],db_type=db_type)
    update_indexes(index_pool, db_type, update=True)
    #
    stock_pool = get_stock_pool()
    print(stock_pool.shape)
    print(stock_pool.head())
    update_stocks(stock_pool, db_type=db_type, update=True)
    #
    # init_table(const.STOCK_DAILY_BASIC[const.TABLE],db_type=db_type)
    update_stock_daily_basic(stock_pool=stock_pool, db_type=db_type,
                             update=True)


def get_return():
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20100101
    df = dbop.create_df(cursor, STOCK_DAY[TABLE], start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600350.SH'",
                        )
    df = dp.proc_stock_d(dp.prepare_stock_d(df))
    print(df.shape)
    return_script(df)


def update_dataset():
    targets = [{"period": 20, "func": "max", "col": "high"},
               {"period": 20, "func": "min", "col": "low"},
               {"period": 20, "func": "avg", "col": ""},
               {"period": 5, "func": "max", "col": "high"},
               {"period": 5, "func": "min", "col": "low"},
               {"period": 5, "func": "avg", "col": ""}, ]
    paras = [("y_l_rise",
              {"pred_period": 20, "is_high": True, "is_clf": False,
               "threshold": 0.2}),
             ("y_l_decline",
              {"pred_period": 20,"is_high": False,"is_clf": False,
               "threshold": 0.2}),
             ("y_l_avg",
              {"pred_period": 20, "is_high": True, "is_clf": False,
               "threshold": 0.2, "target_col": "f20avg_f1mv"}),
             ("y_s_rise",
              {"pred_period": 5,"is_high": True,"is_clf": False,
               "threshold": 0.1}),
             ("y_s_decline",
              {"pred_period": 5, "is_high": False, "is_clf": False,
               "threshold": 0.1}),
             ("y_s_avg",
              {"pred_period": 5, "is_high": True,"is_clf": False,
               "threshold": 0.1,"target_col": "f5avg_f1mv"}), ]

    IO_op.save_dataset_in_hdf5(targets=targets, paras=paras, start_year=2019,
                         start_index=0, end_year=2020, end_index=0,
                         slice_length=12, version="2019-03-06")
    IO_op.save_shuffle_info(update_mode="latest")


def train_model():
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 256)

    cols_category = ["area", "market", "exchange", "is_hs"]
    # ycol1, ycol2, ycol3, ycol4 = "y_l_r", "y_l", "y_l_avg", "y_l_rise"

    reg_params = [
        {"n_estimators": 10, "learning_rate": 2, "num_leaves": 15,
         "max_depth": 8,
         "objective": cus_obj.custom_revenue_obj,
         "min_child_samples": 30, "random_state": 0, },
        {"n_estimators": 10, "learning_rate": 2, "num_leaves": 15,
         "max_depth": 8,
         "objective": cus_obj.custom_revenue2_obj,
         "min_child_samples": 30, "random_state": 0, },
        {"n_estimators": 25, "learning_rate": 0.2, "num_leaves": 31,
         "max_depth": 12,
         "min_child_samples": 30, "random_state": 0, },
        {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 31,
         "max_depth": 12,
         "min_child_samples": 30, "random_state": 0, },
    ]
    objs = [("custom_revenue",
             {"f_revenue": cus_obj.custom_revenue,
              "y_transform": cus_obj.custom_revenu_transform}),
            ("custom_revenue2",
             {"f_revenue": cus_obj.custom_revenue2,
              "y_transform": cus_obj.custom_revenu2_transform}),
            ("l2", {"f_revenue": cus_obj.l2_revenue}),
            ("l2", {"f_revenue": cus_obj.l2_revenue})
            ]
    targets = ["y_l_rise", "y_s_rise",
               "y_l_decline", "y_s_decline",
               "y_l_avg", "y_s_avg",
               "y_l", "y_l_r"]
    #
    layer0 = {}
    for target in targets[:6]:
        layer0.update(
            {obj_type + "_" + target: (lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target})
             for (obj_type, obj_dict), kwargs in
             zip(objs[:2], reg_params[:2])})
    # del layer0["l2_y_l"]

    layer1 = {}
    for target in targets[:6]:
        layer1.update(
            {obj_type + "_" + target: (lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target})
             for (obj_type, obj_dict), kwargs in zip(objs[2:3], reg_params[2:3])})

    layer2 = {}
    for target in targets[-1:]:
        layer2.update(
            {obj_type + "_" + target: (lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target})
             for (obj_type, obj_dict), kwargs in zip(objs[-1:], reg_params[-1:])})
    layers = [layer0, layer1, layer2]

    lgbm_reg_net = ml.RegressorNetwork()
    lgbm_reg_net.insert_multiple_layers(layers)

    paras = {"fit": {"categorical_feature": cols_category}}
    trading_dates = sorted(dbop.get_trading_dates())
    end = str(trading_dates[-20])
    print(end)
    # end = "-".join([end[:4], end[4:6], end[6:]])
    # print(end)
    for i in range(lgbm_reg_net.get_num_layers()):
        X, Y, _ = IO_op.read_hdf5(start="2013-01-01", end=end,
                                  subsample="10-{0}".format(i))
        print(X.info(memory_usage="deep"))
        del X["industry"]

        Y["y_l_r"] = Y.apply(
            lambda r: r["y_l_rise"] if r["y_l_avg"] > 0 else r["y_l_decline"],
            axis=1) * 0.75 + 0.25 * Y["y_l_avg"].values
        print(Y[Y["y_l_avg"].isnull()].shape)
        print(Y[Y["y_l_avg"].isnull()].iloc[:20])

        cond = Y.notnull().all(axis=1)
        X = X[cond]
        Y = Y[cond]

        trading_dates = Y.index.unique().sort_values(ascending=True)
        train_dates = trading_dates[:-21]
        X = X.loc[train_dates]
        Y = Y.loc[train_dates]
        print("Train dates:{0}-{1}".format(min(train_dates), max(train_dates)))
        print(X.shape)

        if i > 0:
            for j in range(i):
                features, _ = lgbm_reg_net.predict_layer(j, X)
                X = pd.concat([X, features], axis=1)

        lgbm_reg_net.fit_layer(i, X, Y, **paras)
        del X, Y

    model_dir = "models"
    model_f_name = "lgbm_reg_net"
    model_path = os.path.join(model_dir, model_f_name + "_{0}".format(
        datetime.datetime.now().strftime("%Y%m%d")))
    with open(model_path, mode="wb") as f:
        pickle.dump(lgbm_reg_net, f)

    return lgbm_reg_net


def predict(model_net):
    X, _, df_other = IO_op.read_hdf5(start="2019-01-01", end="2020-01-01",
                                     # subsample="1-0"
                                     )
    print(X.info(memory_usage="deep"))
    del X["industry"]
    predict_dates = sorted(X.index.unique())[-20:]
    df_all = pd.concat([X, df_other[["code"]]], axis=1).loc[predict_dates]
    X = df_all[df_all.columns.difference(["code"])]
    df_codes = df_all[["code"]]

    base_dir = "predict_results"
    for i in range(model_net.get_num_layers()):
        result = model_net.predict(X, i + 1)
        df = pd.concat([df_codes, result], axis=1)
        f_name = "result_{0}_layer{1}.csv".format(max(predict_dates), i)
        # df.to_csv(os.path.join(base_dir, f_name))
        cols = [col for col in df.columns if col[-4:] != "leaf"]
        df[cols].to_csv(os.path.join(base_dir, "pred_" + f_name))


def daily_job1():
    print("Daily job1 start!")
    collect_data()
    update_dataset()
    print("Daily job1 end!")
    gc.collect()

def daily_job2():
    print("Daily job2 start!")
    model_net = train_model()
    predict(model_net)
    print("Daily job2 end!")
    gc.collect()


def daily_job():
    # daily_job1()
    # daily_job2()
    collect_data()
    get_return()


if __name__ == '__main__':
    # schedule.every().day.at("17:00").do(daily_job1)
    # schedule.every().day.at("18:16").do(daily_job2)
    schedule.every().day.at("17:00").do(daily_job)
    while True:
        schedule.run_pending()
        time.sleep(1)


