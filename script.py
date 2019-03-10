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
import os

import io_operations as IO_op
import ml_model as ml



def find_max_min_point(outer_slope,outer):
    pass



if __name__ == '__main__':
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 256)
    base_dir = "predict_results"

    cols_category = ["area", "market", "exchange", "is_hs"]
    ycol1, ycol2, ycol3, ycol4 = "y_l_r", "y_l", "y_l_avg", "y_l_rise"

    regs = [lgbm.LGBMRegressor(n_estimators=10, learning_rate=2, num_leaves=15,
                               max_depth=8,
                               objective=cus_obj.custom_revenue_obj,
                               min_child_samples=30, random_state=0, ),
        lgbm.LGBMRegressor(n_estimators=10, learning_rate=2, num_leaves=15,
                           max_depth=8, objective=cus_obj.custom_revenue2_obj,
                           min_child_samples=30, random_state=0, ),
        lgbm.LGBMRegressor(n_estimators=25, num_leaves=31, max_depth=12,
                           min_child_samples=30, random_state=0,
                           learning_rate=0.2),
        lgbm.LGBMRegressor(n_estimators=50, num_leaves=31, max_depth=12,
                           min_child_samples=30, random_state=0), ]
    objs = [("custom_revenue", {"f_revenue": cus_obj.custom_revenue,
                                "y_transform": cus_obj.custom_revenu_transform}),
            ("custom_revenue2", {"f_revenue": cus_obj.custom_revenue2,
                                 "y_transform": cus_obj.custom_revenu2_transform}),
            ("l2", {"f_revenue": cus_obj.l2_revenue}),
            ("l2", {"f_revenue": cus_obj.l2_revenue})]
    targets = ["y_l_rise", "y_s_rise", "y_l_decline", "y_s_decline", "y_l_avg",
               "y_s_avg", "y_l", "y_l_r"]
    #
    layer0 = {}
    for target in targets[:6]:
        layer0.update(
            {obj_type + "_" + target: (reg, {**obj_dict, "target": target}) for
             (obj_type, obj_dict), reg in zip(objs[:2], regs[:2])})
    layer1 = {}
    for target in targets[:6]:
        layer1.update(
            {obj_type + "_" + target: (reg, {**obj_dict, "target": target}) for
             (obj_type, obj_dict), reg in zip(objs[2:3], regs[2:3])})
    layer2 = {}
    for target in targets[-1:]:
        layer2.update(
            {obj_type + "_" + target: (reg, {**obj_dict, "target": target}) for
             (obj_type, obj_dict), reg in zip(objs[-1:], regs[-1:])})
    layers = [layer0, layer1, layer2]

    lgbm_reg_net = ml.RegressorNetwork()
    lgbm_reg_net.insert_multiple_layers(layers)

    paras = {"fit": {"categorical_feature": cols_category}}
    for i in range(lgbm_reg_net.get_num_layers()):
        X, Y, _ = IO_op.read_hdf5(start="2013-01-01", end="2019-01-01",
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
        print("Train dates:{0}-{1}".format(min(train_dates),max(train_dates)))
        print(X.shape)

        if i>0:
            for j in range(i):
                features, _ = lgbm_reg_net.predict_layer(j, X)
                X = pd.concat([X, features], axis=1)

        lgbm_reg_net.fit_layer(i,X,Y[ycol1],**paras)
        del X,Y

    X, _, df_other = IO_op.read_hdf5(start="2019-01-01", end="2020-01-01",
                              subsample="1-0")
    print(X.info(memory_usage="deep"))
    del X["industry"]
    predict_dates = sorted(X.index.unique())[-20:]
    df_all = pd.concat([X,df_other[["code"]]],axis=1).loc[predict_dates]
    X = df_all[df_all.columns.difference(["code"])]
    df_codes = df_all[["code"]]

    for i in range(lgbm_reg_net.get_num_layers()):
        result = lgbm_reg_net.predict(X,i+1)
        pd.concat([df_codes,result],axis=1)\
            .to_csv(
            os.path.join(base_dir,
                         "result_{0}_layer{1}.csv".format(max(predict_dates),i)))