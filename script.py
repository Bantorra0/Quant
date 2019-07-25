import datetime
import os
import pickle
import time

import lightgbm.sklearn as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import db_operations as dbop
import customized_obj as cus_obj
import io_operations as IO_op
import data_process as dp
import ml_model as ml
from constants import *


IDX = pd.IndexSlice


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
                    holding_days=20, holding_threshold=0.1,max_days=60,new_high_days_limit=20, is_truncated=True):
    if len(df_single_stock_d[(df_single_stock_d["vol"]>0)
                          & (df_single_stock_d[["open","high","low","close"]].notnull().all(axis=1))]) == 0:
        print(df_single_stock_d.sort_index())
        return None

    # Cleaning input.
    df_single_stock_d = \
        df_single_stock_d[(df_single_stock_d["vol"]>0)
                          & (df_single_stock_d[["open","high","low","close"]].notnull().all(axis=1))].copy()
    df_single_stock_d = df_single_stock_d.sort_index(ascending=True)


    # Result dataframe
    result = pd.Series(index=df_single_stock_d.index)
    # result.index.name = "date"

    # Dataframe for intermediate result(info of holding shares).
    df_tmp = pd.DataFrame(columns=["date","code","open","max","buy_idx","high_idx"]).set_index(["date","code"])
    # df_tmp = pd.DataFrame(columns=["open","max","buy_idx","high_idx"])

    # df_tmp.index.name="date"

    for i in range(1,len(df_single_stock_d.index)):
        curr_dt = df_single_stock_d.index[i]
        prev_dt = df_single_stock_d.index[i-1]

        prev_low, prev_high = df_single_stock_d.loc[prev_dt, ["low", "high"]]
        curr_open, curr_high = df_single_stock_d.loc[curr_dt, ["open", "high"]]

        # Try stop profit next.
        # stop_profit_points = df_tmp["max"] * 0.9
        # stop_profit_points2 = df_tmp["open"] + (df_tmp["max"] - df_tmp["open"]) * (1 - retracement_inc_pct)
        # cond = stop_profit_points > stop_profit_points2
        # stop_profit_points.loc[cond] = stop_profit_points2.loc[cond]

        stop_profit_points = df_tmp["open"] * (1-loss_limit) + (df_tmp["max"] - df_tmp["open"]) * (1 - retracement_inc_pct)
        stop_profit_cond = prev_low <= stop_profit_points
        # stop_profit_idx = df_tmp.index[stop_profit_cond]
        # result = result.append(curr_open / df_tmp.loc[stop_profit_cond]["open"] - 1)
        result.loc[df_tmp.index[stop_profit_cond]] = curr_open
        # df_tmp = df_tmp[~stop_profit_cond]
        # del df_tmp.loc[stop_profit_idx]
        # df_tmp =df_tmp.drop(stop_profit_idx,axis=0)
        df_tmp = df_tmp[~stop_profit_cond]


        # # Try stop loss first.
        # stop_loss_cond = (prev_low <= df_tmp["open"]*(1-retracement))
        # # stop_loss_idx = df_tmp.index[stop_loss_cond]
        # result = result.append(curr_open/df_tmp.loc[stop_loss_cond]["open"]-1)
        # # df_tmp = df_tmp[~stop_loss_cond]
        # # del df_tmp.loc[stop_loss_idx]
        # df_tmp =df_tmp[~stop_loss_cond]


        # # Try to sell if holding for too long.
        # holding_too_long_cond1 = (i-1-df_tmp["idx"]>=holding_days) & (df_tmp["max"]/df_tmp["open"]-1<holding_threshold)
        # # holding_too_long_idx1 = df_tmp.index[holding_too_long_cond1]
        # result = result.append(curr_open/df_tmp.loc[holding_too_long_cond1]["open"]-1)
        # # df_tmp = df_tmp[~holding_too_long_cond1]
        # # del df_tmp.loc[holding_too_long_idx1]
        # # df_tmp=df_tmp.drop(holding_too_long_idx1,axis=0)
        # df_tmp=df_tmp[~holding_too_long_cond1]
        #
        # holding_too_long_cond2 = \
        #     (i - 1 - df_tmp["idx"] >= holding_days*1.5) \
        #     & (prev_close / df_tmp["open"] - 1 < holding_threshold)
        # # holding_too_long_idx2 = df_tmp.index[holding_too_long_cond2]
        # result = result.append(curr_open / df_tmp.loc[holding_too_long_cond2]["open"] - 1)
        # # df_tmp = df_tmp[~holding_too_long_cond2]
        # # del df_tmp.loc[holding_too_long_idx2]
        # # df_tmp =df_tmp.drop(holding_too_long_idx2,axis=0)
        # df_tmp = df_tmp[~holding_too_long_cond2]

        # Sell if max_days is not none and max_days is exceeded.
        if max_days is not None:
            cond = (i - df_tmp["buy_idx"] >= max_days)
            result.loc[df_tmp.index[cond]] = curr_open
            df_tmp = df_tmp[~cond]

        if new_high_days_limit is not None:
            # cond = (prev_high <= df_tmp["max"]) & (i-1-df_tmp["high_idx"]>=new_high_days_limit)
            cond = (i-df_tmp["high_idx"]>=new_high_days_limit)
            result.loc[df_tmp.index[cond]] = curr_open
            df_tmp = df_tmp[~cond]

        # Buy in at the beginning of current date with open price.Add new record.
        df_tmp.loc[prev_dt,:] = list([curr_open,curr_high,i,i])
        # Update max.
        df_tmp.loc[df_tmp["max"]<curr_high,["max","high_idx"]]=[curr_high,i]
        # print(stop_loss_idx)
        # print(stop_profit_idx)
        # print(holding_too_long_idx1)
        # print(holding_too_long_idx2)
        # print(df_tmp)
        # print(result.shape)

    if is_truncated:
        curr_open = df_single_stock_d.loc[df_single_stock_d.index[-1], "open"]
        result.loc[df_tmp.index] = curr_open

    # result = pd.DataFrame(result,columns=["r"])
    # result["code"] = df_single_stock_d["code"].iloc[0]
    # print(result)
    return result.sort_index()


def get_return_rate2(df_single_stock_d: pd.DataFrame, loss_limit=0.1, retracement=0.1, retracement_inc_pct=0.25,
                    holding_days=20, holding_threshold=0.1,max_days=60,new_high_days_limit=20, is_truncated=True):
    # Cleaning input.
    df_single_stock_d = \
        df_single_stock_d[(df_single_stock_d["vol"] > 0)
                          & (df_single_stock_d[["open", "high", "low", "close"]].notnull().all(axis=1))].copy()
    df_single_stock_d = df_single_stock_d.sort_index(ascending=True)

    # result = pd.DataFrame(columns="sell_price")
    result = pd.Series(index=df_single_stock_d.index)
    # result.index.name = "date"

    # Dataframe for intermediate result(info of holding shares).
    df_tmp = pd.DataFrame(columns=["date","code","open","max","buy_idx","high_idx"]).set_index(["date","code"])
    # df_tmp.index.name = "date"

    for i in range(1, len(df_single_stock_d.index)):
        curr_dt = df_single_stock_d.index[i]
        prev_dt = df_single_stock_d.index[i - 1]

        prev_low,prev_high = df_single_stock_d.loc[prev_dt, ["low","high"]]
        curr_open, curr_high = df_single_stock_d.loc[curr_dt, ["open", "high"]]


        # # Try stop loss first.
        # mask = (prev_low <= df_tmp["open"] * (1 - retracement))

        # Try to sell based on retracement.
        stop_profit_points = df_tmp["open"] * (1-loss_limit) + (df_tmp["max"] - df_tmp["open"]) * (1 - retracement_inc_pct)
        # stop_profit_points2 = df_tmp["open"] + (df_tmp["max"] - df_tmp["open"]) * (1 - retracement_inc_pct)
        # cond = stop_profit_points > stop_profit_points2
        # stop_profit_points.loc[cond] = stop_profit_points2.loc[cond]
        # mask |= prev_low <= stop_profit_points
        mask = prev_low <= stop_profit_points

        # Sell if max_days is not none and max_days is exceeded.
        if max_days is not None:
            mask |= (i - df_tmp["buy_idx"] >= max_days)

        if new_high_days_limit is not None:
            # mask |= (prev_high <= df_tmp["max"]) & (i-1-df_tmp["high_idx"]>=new_high_days_limit)
            mask |= (i-df_tmp["high_idx"]>=new_high_days_limit)


        # # Try to sell if holding for too long.
        # mask |= (i - 1 - df_tmp["idx"] >= holding_days) & (
        #             df_tmp["max"] / df_tmp["open"] - 1 < holding_threshold)
        #
        # mask |= \
        #     (i - 1 - df_tmp["idx"] >= holding_days * 1.5) \
        #     & (prev_close / df_tmp["open"] - 1 < holding_threshold)

        # conditions = [stop_loss_cond, stop_profit_cond, holding_too_long_cond1, holding_too_long_cond2]
        # mask = conditions[0]
        # for cond in conditions[1:]:
        #     mask |= cond
        # result = result.append(pd.Series(curr_open,index=df_tmp.index[mask]))
        result.loc[df_tmp.index[mask]] = curr_open
        df_tmp = df_tmp[~mask]

        # Buy in at the beginning of current date with open price.Add new record.
        df_tmp.loc[prev_dt,:] = list([curr_open, curr_high, i, i])
        # Update max.
        df_tmp.loc[df_tmp["max"] < curr_high, ["max","high_idx"]] = [curr_high,i]

    if is_truncated:
        curr_open = df_single_stock_d.loc[df_single_stock_d.index[-1], "open"]
        # result.loc[df_tmp.index] = curr_open
        # result = result.append(pd.Series(curr_open,index=df_tmp))
        result.loc[df_tmp.index] = curr_open

    return result.sort_index()


def get_return_rate3(df_single_stock_d: pd.DataFrame, loss_limit=0.1, retracement=0.1, retracement_inc_pct=0.25,
                    holding_days=20, holding_threshold=0.1,max_days=60,new_high_days_limit=20, is_truncated=True):
    # Cleaning input.
    df_single_stock_d = \
        df_single_stock_d[(df_single_stock_d["vol"] > 0)
                          & (df_single_stock_d[["open", "high", "low", "close"]].notnull().all(axis=1))].copy()
    df_single_stock_d.sort_index(ascending=True, inplace=True)
    index = df_single_stock_d.index
    n = len(df_single_stock_d)

    date_id = np.arange(n)
    # Dataframe for intermediate result(info of holding shares).
    df_tmp = pd.DataFrame(columns=["open","max"],index=index)
    df_tmp.iloc[:-1] = df_single_stock_d[["open","high"]].values[1:]
    df_tmp["idx"] = date_id
    df_tmp["max_idx"] = df_tmp["idx"]+1 # Buy in next day, so here we increment 1.
    df_tmp["is_selled"] = False

    cnt = 1
    # columns = ["open","high","low","close","avg"]
    columns = ["open", "low","high"]
    # col_idx = {col:i for i,col in enumerate(columns)}
    mask = pd.Series(index=index)
    # slice1 = slice(0,n - cnt)
    # slice2 = slice(cnt,None)
    # df_curr = pd.DataFrame(columns=columns, index=index)
    # df_curr.loc[index[slice1], columns] = df_single_stock_d.loc[index[slice2], columns].values
    # df_curr.loc[index[slice1], "idx"] = date_id[slice2]
    while df_tmp["is_selled"].all()!=True:
        df_curr = pd.DataFrame(columns=columns, index=index)
        slice1 = slice(0, n - cnt)
        slice2 = slice(cnt, None)
        df_curr.loc[index[slice1], columns] = df_single_stock_d.loc[index[slice2], columns].values
        df_curr.loc[index[slice1], "idx"] = date_id[slice2]
        # a_curr = df_single_stock_d.values

        # print("--",df_tmp[mask])
        mask &= ((df_tmp["is_selled"] == False) & df_curr["open"].notnull())
        df_tmp.loc[mask, "sell_price"] = df_curr.loc[mask, "open"].values
        df_tmp.loc[mask, "is_selled"] = True

        if df_tmp["is_selled"].iloc[0:n-cnt-1].all():
            # print("all selled")
            break

        # mask &= ((df_tmp.loc[index1,"is_selled"] == False) & np.isnan(a_curr[col_idx["open"]]))
        # mask_idx = index[np.nonzero(mask)]
        # df_tmp.loc[mask_idx, "sell_price"] = a_curr[col_idx["open"]]
        # df_tmp.loc[mask_idx, "is_selled"] = True

        cond = (df_tmp["is_selled"] == False) & (df_tmp["max"] < df_curr["high"])
        df_tmp.loc[cond, ["max", "max_idx"]] = df_curr.loc[cond, ["high", "idx"]].values

        # cond = (df_tmp.loc[index1,"is_selled"] == False) & (df_tmp.loc[index1,"max"] < a_curr[col_idx["high"]])
        # cond_idx = index[np.nonzero(cond)]
        # df_tmp.loc[cond_idx, ["max", "max_idx"]] = a_curr[[col_idx["high"], col_idx["idx"]]]

        # print(cnt,df_tmp.shape,df_tmp["is_selled"].sum())
        # print(df_tmp[mask])


        # df_prev = df_curr
        # a_prev = a_curr
        cnt+=1

        stop_profit_points = df_tmp["open"] * (1-loss_limit) + (df_tmp["max"] - df_tmp["open"]) * (1 - retracement_inc_pct)
        # print(pd.concat([df_prev,stop_profit_points],axis=1).iloc[:10])
        mask = df_curr["low"] <= stop_profit_points

        # mask = a_curr["low"] <= stop_profit_points.loc[index1]
        # print((mask & (df_tmp["is_selled"]==False)).sum())
        # print(df_tmp[mask & (df_tmp["is_selled"]==False)])
        # print(df_prev[mask & (df_tmp["is_selled"]==False)])

        # Sell if max_days is not none and max_days is exceeded.
        if max_days is not None:
            # print(df_tmp[(df_tmp["is_selled"]==False)&(df_prev["idx"] - df_tmp["buy_idx"] >= max_days)])
            mask |= (df_curr["idx"] - df_tmp["idx"] >= max_days)

        if new_high_days_limit is not None:
            mask |= (df_curr["idx"]+1 - df_tmp["max_idx"] >= new_high_days_limit) # Buy in next day, so here we increment 1.

    if is_truncated:
        curr_open = df_single_stock_d["open"].iloc[-1]
        mask0 = (df_tmp["is_selled"]==False) & (df_tmp["idx"]<n-1)
        df_tmp.loc[mask0,"sell_price"] = curr_open
        df_tmp.loc[mask0,"is_selled"] = True

    return df_tmp


def get_return_rate_batch(df_stock_d: pd.DataFrame, loss_limit=0.1, retracement=0.1, retracement_inc_pct=0.25,
                          max_days=60, new_high_days_limit=20, is_truncated=True):
    # Cleaning input.
    df_stock_d = \
        df_stock_d[(df_stock_d["vol"] > 0)
                   & (df_stock_d[["open", "high", "low", "close"]].notnull().all(axis=1))].copy()

    df_single_stock_d_list = []
    a_trunc_list = []
    for code,df_single_stock_d in df_stock_d.groupby("code"):
        k = len(df_single_stock_d)
        df_single_stock_d=df_single_stock_d.sort_index(ascending=True)
        df_single_stock_d["idx"] = np.arange(k)
        df_single_stock_d_list.append(df_single_stock_d)

        trunc_open = np.ones(k)*df_single_stock_d["open"].iloc[-1]
        trunc_open[-1] = np.nan
        a_trunc_list.append(trunc_open)

    print(len(df_single_stock_d_list))
    df_stock_d = pd.concat(df_single_stock_d_list,axis=0)
    origin_index = df_stock_d.index
    df_stock_d.reset_index(level="code", inplace=True)
    df_stock_d.set_index(["code", "idx"], inplace=True)
    index = df_stock_d.index
    a_trunc_open = np.concatenate(a_trunc_list,axis=0)

    # Dataframe for intermediate result(info of holding shares).
    df_tmp = df_stock_d[["open","high"]].rename(columns={"high":"max"})
    df_tmp["idx"] = df_tmp.index.get_level_values("idx")-1
    df_tmp["max_idx"] = df_tmp["idx"] + 1 # Buy in next day, so here we increment 1.
    df_tmp["is_selled"] = False
    df_tmp.reset_index(level="code", inplace=True)
    df_tmp.set_index(["code", "idx"], inplace=True)
    df_tmp = df_tmp.reindex(index=index)

    columns = ["open", "low","high","close"]
    df_curr = df_stock_d[columns].copy()
    df_curr["idx"] = index.get_level_values("idx")
    df_curr = df_curr.groupby(level="code").shift(-1)
    mask = pd.Series(False, index=index)
    while df_tmp["is_selled"].all()!=True:
        df_prev = df_curr
        df_curr = df_curr.groupby(level="code").shift(-1)

        if df_tmp.loc[df_curr["open"].notnull(),"is_selled"].all():
            break

        # Sell shares given mask.
        mask &= ~((df_curr["low"]==df_curr["high"])&(df_curr["open"]<df_prev["close"])) # 排除跌停板卖不出的情况
        mask &= ((df_tmp["is_selled"] == False) & df_curr["open"].notnull())
        df_tmp.loc[mask, "sell_price"] = df_curr.loc[mask, "open"].values
        df_tmp.loc[mask, "is_selled"] = True

        cond = (df_tmp["is_selled"] == False) & (df_tmp["max"] < df_curr["high"])
        df_tmp.loc[cond, ["max", "max_idx"]] = df_curr.loc[cond, ["high", "idx"]].values

        stop_profit_points = df_tmp["open"] * (1-loss_limit) + (df_tmp["max"] - df_tmp["open"]) * (1 - retracement_inc_pct)
        mask = df_curr["low"] <= stop_profit_points

        # Sell if max_days is not none and max_days is exceeded.
        if max_days is not None:
            # print(df_tmp[(df_tmp["is_selled"]==False)&(df_prev["idx"] - df_tmp["buy_idx"] >= max_days)])
            mask |= (df_curr["idx"] - df_tmp.index.get_level_values("idx") >= max_days)

        if new_high_days_limit is not None:
            mask |= (df_curr["idx"]+1 - df_tmp["max_idx"] >= new_high_days_limit) # Buy in next day, so here we increment 1.

    if is_truncated:
        mask0 = (df_tmp["is_selled"]==False) & (~np.isnan(a_trunc_open))
        df_tmp.loc[mask0, "sell_price"] = a_trunc_open[mask0]
        df_tmp.loc[mask0, "is_selled"] = True

    df_tmp.index = origin_index
    return df_tmp


def get_return_rate_batch2(df_stock_d: pd.DataFrame, loss_limit=0.1, retracement=0.1, retracement_inc_pct=0.25,
                           holding_days=20, holding_threshold=0.1, max_days=60, new_high_days_limit=20, is_truncated=True):
    # Cleaning input.
    df_stock_d = \
        df_stock_d[(df_stock_d["vol"] > 0)
                   & (df_stock_d[["open", "high", "low", "close"]].notnull().all(axis=1))].copy()

    df_single_stock_d_list = []
    a_trunc_list = []
    for code,df_single_stock_d in df_stock_d.groupby("code"):
        k = len(df_single_stock_d)
        df_single_stock_d=df_single_stock_d.sort_index(ascending=True)
        df_single_stock_d["idx"] = np.arange(k)
        df_single_stock_d_list.append(df_single_stock_d)

        trunc_open = np.ones(k)*df_single_stock_d["open"].iloc[-1]
        trunc_open[-1] = np.nan
        a_trunc_list.append(trunc_open)

    df_stock_d = pd.concat(df_single_stock_d_list,axis=0)
    origin_index = df_stock_d.index
    df_stock_d.reset_index(level="code", inplace=True)
    df_stock_d.set_index(["code", "idx"], inplace=True)
    index = df_stock_d.index
    a_trunc_open = np.concatenate(a_trunc_list,axis=0)

    # Dataframe for intermediate result(info of holding shares).
    df_tmp = df_stock_d[["open","high"]].rename(columns={"high":"max"})
    df_tmp["idx"] = df_tmp.index.get_level_values("idx")-1
    df_tmp["max_idx"] = df_tmp["idx"] + 1 # Buy in next day, so here we increment 1.
    df_tmp["is_selled"] = False
    df_tmp.reset_index(level="code", inplace=True)
    df_tmp.set_index(["code", "idx"], inplace=True)
    df_tmp = df_tmp.reindex(index=index)

    columns = ["open", "low", "high", "close"]
    df_curr = df_stock_d[columns].copy()
    df_curr["idx"] = index.get_level_values("idx")
    mask = pd.Series(False,index=index)
    while df_tmp["is_selled"].all()!=True:
        df_curr["idx0"] = df_curr.index.get_level_values("idx") - 1
        df_curr.reset_index(level="code", inplace=True)
        df_curr.set_index(["code", "idx0"],inplace=True)
        df_curr.index.rename(["code","idx"],inplace=True)
        df_curr = df_curr.reindex(index=index)
        # df_curr = df_curr.groupby(level="code").shift(-1)


        if df_tmp.loc[df_curr["open"].notnull(),"is_selled"].all():
            # print("all selled")
            break

        # print("--",df_tmp[mask])
        mask &= ((df_tmp["is_selled"] == False) & df_curr["open"].notnull())
        df_tmp.loc[mask, "sell_price"] = df_curr.loc[mask, "open"].values
        df_tmp.loc[mask, "is_selled"] = True

        cond = (df_tmp["is_selled"] == False) & (df_tmp["max"] < df_curr["high"])
        df_tmp.loc[cond, ["max", "max_idx"]] = df_curr.loc[cond, ["high", "idx"]].values

        stop_profit_points = df_tmp["open"] * (1-loss_limit) + (df_tmp["max"] - df_tmp["open"]) * (1 - retracement_inc_pct)
        mask = df_curr["low"] <= stop_profit_points

        # Sell if max_days is not none and max_days is exceeded.
        if max_days is not None:
            # print(df_tmp[(df_tmp["is_selled"]==False)&(df_prev["idx"] - df_tmp["buy_idx"] >= max_days)])
            mask |= (df_curr["idx"] - df_tmp.index.get_level_values("idx") >= max_days)

        if new_high_days_limit is not None:
            mask |= (df_curr["idx"]+1 - df_tmp["max_idx"] >= new_high_days_limit) # Buy in next day, so here we increment 1.

    if is_truncated:
        mask0 = (df_tmp["is_selled"]==False) & (~np.isnan(a_trunc_open))
        df_tmp.loc[mask0, "sell_price"] = a_trunc_open[mask0]
        df_tmp.loc[mask0, "is_selled"] = True

    df_tmp.index = origin_index
    return df_tmp


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


def test_api():
    import collect
    api = collect._init_api()
    df = api.daily_basic(ts_code="600352.SH", start_date="20190101")
    pd.set_option("display.max_columns", 20)
    print(df)


def test_get_return_rate2():
    cursor = dbop.connect_db("sqlite3").cursor()
    df = dbop.create_df(cursor, STOCK_DAY[TABLE], "2013-01-01")
    print(df.shape)
    df = df[df["code"] == '600352.SH'].set_index("date")
    print(df.shape)

    t0 = time.time()
    r1 = get_return_rate(df)
    print("t1:", time.time() - t0)
    t0 = time.time()
    r2 = get_return_rate2(df)
    print("t2:", time.time() - t0)
    print((r1.dropna() == r2.dropna()).all())
    df = pd.concat([r1, r2], axis=1)
    print(df[df[0] != df[1]])
    print(r1)
    print(r2)


def test_get_return_rate3():
    df_syn = pd.DataFrame(np.array(list(range(100, 150)) + list(range(150, 110, -1))).reshape(-1, 1), columns=["open"])
    df_syn["close"] = df_syn["open"]
    df_syn["high"] = df_syn["open"] + 1
    df_syn["low"] = df_syn["open"] - 1
    df_syn["vol"] = 100
    df_syn["date"] = np.arange(len(df_syn))
    df_syn["code"] = "600352.SH"
    df_syn.set_index(["date","code"],inplace=True)
    t0 = time.time()
    r3 = get_return_rate3(df_syn)
    print(time.time()-t0)
    r1 = get_return_rate(df_syn)
    print(time.time()-t0)

    # print((r1.dropna() == r3.dropna()).all())
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 15)
    df = pd.concat([df_syn, r1, r3], axis=1)
    print(df)
    print(df[df[0] != df["sell_price"]][[0, "sell_price"]])


def test_get_return_rate_batch():
    df_syn = pd.DataFrame(np.array(list(range(100, 150)) + list(range(150, 110, -1))).reshape(-1, 1), columns=["open"])
    df_syn["close"] = df_syn["open"]
    df_syn["high"] = df_syn["open"] + 1
    df_syn["low"] = df_syn["open"] - 1
    df_syn["vol"] = 100
    df_syn["date"] = np.arange(len(df_syn))
    df_syn["code"] = "600352.SH"
    df_syn.set_index(["date","code"],inplace=True)
    t0 = time.time()
    r3 = get_return_rate_batch(df_syn)
    print(time.time()-t0)
    r1 = get_return_rate(df_syn)
    print(time.time()-t0)

    # print((r1.dropna() == r3.dropna()).all())
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 15)
    df = pd.concat([df_syn, r1, r3], axis=1)
    print(df)
    print(df[df[0] != df["sell_price"]][[0, "sell_price"]])


def test_get_return():
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20180101
    df = dbop.create_df(cursor, STOCK_DAY[TABLE],
                        start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600350.SH'",
                        )
    df = dp.proc_stock_d(dp.prepare_stock_d(df))

    print(df.shape)
    t0 = time.time()
    r_batch = get_return_rate_batch(df)
    print(time.time() - t0)
    t0 = time.time()
    r_batch2 = get_return_rate_batch2(df)
    print(time.time() - t0)
    print(r_batch[r_batch["sell_price"] != r_batch2["sell_price"]])
    # r_list = []
    # for code, group in df.groupby(level="code"):
    #     print(time.time() - t0)
    #     r_list.append(get_return_rate3(group))
    # print(time.time()-t0)
    # r3 = pd.concat(r_list,axis=0)
    # # result = pd.concat([r_batch,r3],axis=1)
    # print(r_batch[r_batch["sell_price"]!=r3["sell_price"]])
    # print(r3[r_batch["sell_price"]!= r3["sell_price"]])

    # r3 = get_return_rate3(df)
    # print(time.time()-t0)
    # r1 = get_return_rate(df)
    # print(time.time()-t0)
    # # print((r1.dropna() == r3.dropna()).all())
    #
    # df = pd.concat([r1, r3],axis=1)
    # # print(df)
    # print(df[df[0]!=df["sell_price"]].dropna())

    # test_get_return_rate_batch()
    # test_get_return_rate3()


def assess_feature_test():
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20140101
    df = dbop.create_df(cursor, STOCK_DAILY_BASIC[TABLE],
                        start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600350.SH'",
                        )
    df = dp.prepare_stock_d_basic(df).drop_duplicates()

    df_r = pd.read_parquet(r"database\return_10%_25%_60_20")
    df_r.sort_index(inplace=True)
    print(df_r.info(memory_usage="deep"))
    print(df_r.head(5))
    df_r["r"] = (df_r["sell_price"] / df_r["open"] - 1) * 100

    result = ml.assess_feature3(df[df.columns.difference(["close"])], df_r["r"], 50,plot=True)
    pd.set_option("display.max_columns",10)
    print(result)
    import matplotlib.pyplot as plt
    plt.show()





def feature_explore():
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 20)

    df_r = pd.read_parquet(r"database\return_10%_25%_60_20")
    df_r.sort_index(inplace=True)
    print(df_r.info(memory_usage="deep"))
    print(df_r.head(5))
    df_r["r"] = (df_r["sell_price"] / df_r["open"] - 1) * 100

    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20120101
    df_d_basic = dbop.create_df(cursor, STOCK_DAILY_BASIC[TABLE],
                                start=start,
                                # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                                # where_clause="code='600350.SH'",
                                )
    df_d_basic = dp.prepare_stock_d_basic(df_d_basic).drop_duplicates()

    df_d = dbop.create_df(cursor, STOCK_DAY[TABLE],
                                start=start,
                                # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                                # where_clause="code='600350.SH'",
                                )
    df_d = dp.proc_stock_d(dp.prepare_stock_d(df_d))

    result = ml.assess_feature3(df_d_basic[df_d_basic.columns.difference(["close"])], df_r["r"], 50,plot=True)
    print(result)
    plt.show()


def explore_simple_features():
    import lightgbm as lgbm
    pd.set_option("display.max_columns",20)
    pd.set_option("display.max_rows", 100)
    store = pd.HDFStore("dataset")
    X = store["X"]
    y = store["y"]
    dates = X.index.get_level_values("date")

    train_mask = dates <= "2017-10-31"
    test_mask = dates >= "2018-01-01"
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    print(X_train.shape, X_test.shape)

    reg = lgbm.LGBMRegressor(num_leaves=31,max_depth=12, learning_rate=0.4,
                             n_estimators=5,
                             min_child_samples=100)

    reg.fit(X_train,y_train)
    reg.score(X_test,y_test)
    y_pred = reg.predict(X_test)
    y_pred = pd.Series(y_pred,index=y_test.index,name="y_pred")
    y_test.name="y"
    Y = pd.concat([y_test,y_pred],axis=1)
    l = []
    l.append(np.arange(5,40,5))
    l.append(np.arange(40,100,20))
    l.append(np.arange(100,200,50))
    l.append(np.array([200,float("inf")]))
    positive_bins = np.concatenate(l)
    bins = list(reversed(list(positive_bins*-1)))+[0]+list(positive_bins)
    Y["bin"] = pd.cut(Y["y_pred"],bins=bins)
    Y.groupby("bin").agg({"bin":"size","y_pred":["mean","median"],"y":["mean",
                                                               "median"]})



if __name__ == '__main__':
    # test_get_return()
    # assess_feature_test()

    from feature_engineering import *

    df_r = pd.read_parquet(r"database\return_10%_25%_60_20")
    df_r.sort_index(inplace=True)
    print(df_r.info(memory_usage="deep"))
    print(df_r.head(5))
    df_r["r"] = (df_r["sell_price"] / df_r["open"] - 1) * 100

    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20120101
    df_d_basic = dbop.create_df(cursor, STOCK_DAILY_BASIC[TABLE], start=start,
                                # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                                # where_clause="code='600350.SH'",
                                )
    df_d_basic = dp.prepare_stock_d_basic(df_d_basic).drop_duplicates()

    df_d = dbop.create_df(cursor, STOCK_DAY[TABLE], start=start,
                          # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                          # where_clause="code='600350.SH'",
                          )
    df_d = dp.proc_stock_d(dp.prepare_stock_d(df_d))

    features = pd.DataFrame()

    intervals = [5, 10, 20, 30, 40, 60, 120, 250]
    for days in intervals:
        tmp = groupby_rolling(df_d, level="code", window=days,
                              ops={"low": "min"})
        features["close/p{}min_low-1".format(days)] = df_d["close"] / tmp[
            "low"] - 1

    kma = pd.DataFrame()
    for days in intervals:
        tmp = k_MA_batch(days, df_d)
        col = tmp.columns[0]
        kma[col] = tmp[col]

    for col in kma.columns:
        features["close/{}-1".format(col)] = df_d["close"] / kma[col] - 1

    for col in kma.columns[1:]:
        features["{0}/{1}-1".format(kma.columns[0], col)] = kma[kma.columns[
            0]] / kma[col] - 1

    #
    df_d_basic["pb*pe_ttm"] = df_d_basic["pb"] * df_d_basic["pe_ttm"]
    df_d_basic["pb*pe"] = df_d_basic["pb"] * df_d_basic["pe"]
    df_d_basic["peg"] = df_d_basic["pe_ttm"] / np.where(((df_d_basic["pe"] /
                                                          df_d_basic[
                                                              "pe_ttm"] - 1) * 100 / (
                                                                     df_d_basic.index.get_level_values(
                                                                         "date").quarter - 1) * 4 >= 10) & (
                                                                    df_d_basic.index.get_level_values(
                                                                        "date").quarter > 1),
                                                        (df_d_basic["pe"] /
                                                         df_d_basic[
                                                             "pe_ttm"] - 1) * 100 / (
                                                                    df_d_basic.index.get_level_values(
                                                                        "date").quarter - 1) * 4,
                                                        1)

    features = pd.concat([features, df_d_basic], axis=1)
    print("Features generated!")
    del df_d_basic, kma
    df_d = df_d[[col for col in df_d.columns if "0" not in col]]
    y = df_r["r"].reindex(features.index)
    X = features[y.notnull()]
    y = y.dropna()
    features = features.astype("float32")
    y = y.astype("float32")
    store = pd.HDFStore("dataset")
    store["X"] = X
    store["y"] = y
    store.close()














