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
import pickle

import io_operations as IO_op
import ml_model as ml



def find_max_min_point(outer_slope,outer):
    pass


def tidyup():
    base_dir = "predict_results"
    for f in os.listdir(base_dir):
        if f[:4] == "pred":
            continue
        df = pd.read_csv(os.path.join(base_dir, f))
        cols = [col for col in df.columns if col[-4:] != "leaf"]
        df[cols].to_csv(os.path.join(base_dir, "pred_" + f), index=False)


def get_return_rate(df_single_stock_d:pd.DataFrame, loss_limit=0.1, retracement=0.1, retracement_inc_pct=0.25,
                    holding_days=20, holding_threshold=0.1):
    # Cleaning input.
    df_single_stock_d = \
        df_single_stock_d[(df_single_stock_d["vol"]>0)
                          & (df_single_stock_d[["open","high","low","close"]].notnull().all(axis=1))].copy()
    df_single_stock_d = df_single_stock_d.sort_index(ascending=True)

    # Result dataframe
    result = pd.Series()
    result.index.name = "date"

    # Dataframe for intermediate result(info of holding shares).
    df_tmp = pd.DataFrame(columns=["open","max","idx"])
    df_tmp.index.name="date"

    for i in range(1,len(df_single_stock_d.index)):
        print(i)
        curr_dt = df_single_stock_d.index[i]
        prev_dt = df_single_stock_d.index[i-11]

        prev_low = df_single_stock_d.loc[prev_dt,"low"]
        prev_close = df_single_stock_d.loc[prev_dt, "close"]
        curr_open = df_single_stock_d.loc[curr_dt,"open"]

        # Try stop loss first.
        stop_loss_cond = prev_low <= df_tmp["open"]*(1-retracement)
        stop_loss_idx = df_tmp.index[stop_loss_cond]
        result.loc[stop_loss_idx] = curr_open/df_tmp.loc[stop_loss_idx]["open"]-1
        df_tmp = df_tmp[~stop_loss_cond]

        # Try stop profit next.
        stop_profit_points = df_tmp["max"]*0.9
        stop_profit_points2 = df_tmp["open"]+(df_tmp["max"]-df_tmp["open"])*(1 - retracement_inc_pct)
        idx = stop_profit_points.index[stop_profit_points>stop_profit_points2]
        stop_profit_points.loc[idx] = stop_profit_points2.loc[idx]
        stop_profit_cond = prev_low <= stop_profit_points
        stop_profit_idx = df_tmp.index[stop_profit_cond]
        result.loc[stop_profit_idx] = curr_open/df_tmp.loc[stop_profit_idx]["open"]-1
        df_tmp = df_tmp[~stop_profit_cond]

        # Try to sell if holding for too long.
        holding_too_long_cond1 = (i-1-df_tmp["idx"]>=holding_days) & (df_tmp["max"]/df_tmp["open"]-1<holding_threshold)
        holding_too_long_idx1 = df_tmp.index[holding_too_long_cond1]
        result.loc[holding_too_long_idx1] = curr_open/df_tmp.loc[holding_too_long_idx1]["open"]-1
        df_tmp = df_tmp[~holding_too_long_cond1]

        holding_too_long_cond2 = \
            (i - 1 - df_tmp["idx"] >= holding_days*1.5) \
            & (prev_close / df_tmp["open"] - 1 < holding_threshold)
        holding_too_long_idx2 = df_tmp.index[holding_too_long_cond2]
        result.loc[holding_too_long_idx2] = curr_open / df_tmp.loc[holding_too_long_idx2]["open"] - 1
        df_tmp = df_tmp[~holding_too_long_cond2]

        # Buy in at the beginning of current date with open price.Add new record.
        open,high = df_single_stock_d.loc[curr_dt,["open","high"]]
        df_tmp.loc[prev_dt] = list([open,high,i-1])
        # Update max.
        df_tmp.loc[df_tmp.index[df_tmp["max"]<high],"max"]=high
    return result


if __name__ == '__main__':
    # pd.set_option("display.max_columns", 10)
    # pd.set_option("display.max_rows", 256)
    # base_dir = "predict_results"
    #
    # cols_category = ["area", "market", "exchange", "is_hs"]
    # ycol1, ycol2, ycol3, ycol4 = "y_l_r", "y_l", "y_l_avg", "y_l_rise"
    #
    # regs = [lgbm.LGBMRegressor(n_estimators=10, learning_rate=2, num_leaves=15,
    #                            max_depth=8,
    #                            objective=cus_obj.custom_revenue_obj,
    #                            min_child_samples=30, random_state=0, ),
    #     lgbm.LGBMRegressor(n_estimators=10, learning_rate=2, num_leaves=15,
    #                        max_depth=8, objective=cus_obj.custom_revenue2_obj,
    #                        min_child_samples=30, random_state=0, ),
    #     lgbm.LGBMRegressor(n_estimators=25, num_leaves=31, max_depth=12,
    #                        min_child_samples=30, random_state=0,
    #                        learning_rate=0.2),
    #     lgbm.LGBMRegressor(n_estimators=50, num_leaves=31, max_depth=12,
    #                        min_child_samples=30, random_state=0), ]
    # objs = [("custom_revenue", {"f_revenue": cus_obj.custom_revenue,
    #                             "y_transform": cus_obj.custom_revenu_transform}),
    #         ("custom_revenue2", {"f_revenue": cus_obj.custom_revenue2,
    #                              "y_transform": cus_obj.custom_revenu2_transform}),
    #         ("l2", {"f_revenue": cus_obj.l2_revenue}),
    #         ("l2", {"f_revenue": cus_obj.l2_revenue})]
    # targets = ["y_l_rise", "y_s_rise", "y_l_decline", "y_s_decline", "y_l_avg",
    #            "y_s_avg", "y_l", "y_l_r"]
    # #
    # layer0 = {}
    # for target in targets[:6]:
    #     layer0.update(
    #         {obj_type + "_" + target: (reg, {**obj_dict, "target": target}) for
    #          (obj_type, obj_dict), reg in zip(objs[:2], regs[:2])})
    # layer1 = {}
    # for target in targets[:6]:
    #     layer1.update(
    #         {obj_type + "_" + target: (reg, {**obj_dict, "target": target}) for
    #          (obj_type, obj_dict), reg in zip(objs[2:3], regs[2:3])})
    # layer2 = {}
    # for target in targets[-1:]:
    #     layer2.update(
    #         {obj_type + "_" + target: (reg, {**obj_dict, "target": target}) for
    #          (obj_type, obj_dict), reg in zip(objs[-1:], regs[-1:])})
    # layers = [layer0, layer1, layer2]
    #
    # lgbm_reg_net = ml.RegressorNetwork()
    # lgbm_reg_net.insert_multiple_layers(layers)
    #
    # paras = {"fit": {"categorical_feature": cols_category}}
    # for i in range(lgbm_reg_net.get_num_layers()):
    #     X, Y, _ = IO_op.read_hdf5(start="2013-01-01", end="2019-01-01",
    #                               subsample="10-{0}".format(i))
    #     print(X.info(memory_usage="deep"))
    #     del X["industry"]
    #
    #     Y["y_l_r"] = Y.apply(
    #         lambda r: r["y_l_rise"] if r["y_l_avg"] > 0 else r["y_l_decline"],
    #         axis=1) * 0.75 + 0.25 * Y["y_l_avg"].values
    #     print(Y[Y["y_l_avg"].isnull()].shape)
    #     print(Y[Y["y_l_avg"].isnull()].iloc[:20])
    #
    #     cond = Y.notnull().all(axis=1)
    #     X = X[cond]
    #     Y = Y[cond]
    #
    #     trading_dates = Y.index.unique().sort_values(ascending=True)
    #     train_dates = trading_dates[:-21]
    #     X = X.loc[train_dates]
    #     Y = Y.loc[train_dates]
    #     print("Train dates:{0}-{1}".format(min(train_dates),max(train_dates)))
    #     print(X.shape)
    #
    #     if i>0:
    #         for j in range(i):
    #             features, _ = lgbm_reg_net.predict_layer(j, X)
    #             X = pd.concat([X, features], axis=1)
    #
    #     lgbm_reg_net.fit_layer(i,X,Y[ycol1],**paras)
    #     del X,Y
    #
    # model_dir = "models"
    # model_f_name = "lgbm_reg_net"
    # model_path = os.path.join(model_dir,model_f_name+"_{0}".format(datetime.datetime.now().strftime("%Y%m%d")))
    # with open(model_path,mode="wb") as f:
    #     pickle.dump(lgbm_reg_net,f)
    #
    # X, _, df_other = IO_op.read_hdf5(start="2019-01-01", end="2020-01-01",
    #                           subsample="1-0")
    # print(X.info(memory_usage="deep"))
    # del X["industry"]
    # predict_dates = sorted(X.index.unique())[-20:]
    # df_all = pd.concat([X,df_other[["code"]]],axis=1).loc[predict_dates]
    # X = df_all[df_all.columns.difference(["code"])]
    # df_codes = df_all[["code"]]
    #
    # for i in range(lgbm_reg_net.get_num_layers()):
    #     result = lgbm_reg_net.predict(X,i+1)
    #     df = pd.concat([df_codes,result],axis=1)
    #     f_name = "result_{0}_layer{1}.csv".format(max(predict_dates),i)
    #     df.to_csv(os.path.join(base_dir,f_name))
    #     cols = [col for col in df.columns if col[-4:] != "leaf"]
    #     df[cols].to_csv(os.path.join(base_dir, "pred_" + f_name))

    df = pd.DataFrame(np.random.normal(10,2,size=(100,4)),columns=["open","high","low","close"])
    df["vol"]=100
    get_return_rate(df)

