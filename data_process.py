import multiprocessing as mp
import queue
import time

import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc

import constants as const
import collect
import feature_engineering as FE

IDX = pd.IndexSlice


def prepare_stock_d(df_stck_d):
    # if df_stck_d["date"].dtypes!=int:
    #     df_stck_d["date"] = df_stck_d["date"].apply(lambda x:x.replace("-", "")).astype(int)
    df_stck_d["date"] = pd.to_datetime(df_stck_d["date"],format="%Y%m%d")
    df_stck_d = df_stck_d.set_index(["code","date"]).sort_index()

    return df_stck_d[["open", "high", "low", "close", "vol", "amt", "adj_factor"]]


def prepare_index_d(df_idx_d):
    # if df_idx_d["date"].dtype=="object":
    #     df_idx_d["date"]=df_idx_d["date"].apply(lambda x:x.replace("-", "")).astype(int)
    df_idx_d["date"] = pd.to_datetime(df_idx_d["date"],format="%Y%m%d")

    df_idx_d["avg"] = df_idx_d["amt"]/df_idx_d["vol"] * 10
    df_idx_d = df_idx_d.set_index(["code","date"]).sort_index()
    return df_idx_d[["open", "high", "low", "close", "vol", "amt"]]


def prepare_each_stock(df_stock_d, qfq_type="hfq"):
    if qfq_type and qfq_type not in ["hfq","qfq"]:
        raise ValueError("qfq_type {} is not supported".format(qfq_type))

    df_stock_d = df_stock_d.copy()
    # Calculate stocks' avg day prices.
    # vol's unit is "手", while amt's unit is "1000yuan", so 10 is multiplied.
    df_stock_d["avg"] = (df_stock_d["amt"] / df_stock_d["vol"] * 10)
    # 如果vol=0（停牌），则上述计算结果为inf，取收盘价。
    # 此处假设停牌期间已完成数据填充，按vol=0，amt=0，其他价格按前一天（停牌前最后一天）收盘价算。
    mask0 = df_stock_d["vol"]==0
    df_stock_d.loc[mask0,"avg"] = df_stock_d.loc[mask0,"close"]
    # print(df_stock_d.loc[df_stock_d.index[df_stock_d["avg"].isnull()]])

    fq_cols = ["open", "high", "low", "close","avg","vol"]

    # 保存原始数据到新的列。
    for col in fq_cols:
        df_stock_d[col + "0"] = df_stock_d[col]

    # 前复权
    if qfq_type=="qfq":
        fq_factor = df_stock_d["adj_factor"]/ df_stock_d["adj_factor"].iloc[0]
    else:
        fq_factor = df_stock_d["adj_factor"]

    # print(fq_factor.shape)
    print(fq_factor.shape)
    fq_factor = np.array(fq_factor).reshape(-1, 1) * np.ones((1, len(fq_cols)))
    print(fq_factor.shape)

    # Deal with open,high,low,close.
    df_stock_d.loc[:, fq_cols[:5]] = df_stock_d[fq_cols[:5]] * fq_factor[:, :5]
    # Deal with vol.
    df_stock_d.loc[:, fq_cols[5]] = df_stock_d[fq_cols[5]] / fq_factor[:, 0]

    return df_stock_d


def prepare_stock_d_basic(df_stock_d_basic:pd.DataFrame):
    df_stock_d_basic["date"] = pd.to_datetime(df_stock_d_basic["date"],format="%Y%m%d")
    df_stock_d_basic.set_index(["code","date"],inplace=True)
    return df_stock_d_basic


def proc_stock_d(df_stock_d, qfq_type="hfq"):
    if qfq_type and qfq_type not in ["hfq"]:
        raise ValueError("qfq_type {} is not supported".format(qfq_type))

    df_stock_d = df_stock_d.copy()
    # Calculate stocks' avg day prices.
    # vol's unit is "手", while amt's unit is "1000yuan", so 10 is multiplied.
    df_stock_d["avg"] = (df_stock_d["amt"] / df_stock_d["vol"] * 10)
    # 如果vol=0（停牌），则上述计算结果为inf，取收盘价。
    # 此处假设停牌期间已完成数据填充，按vol=0，amt=0，其他价格按前一天（停牌前最后一天）收盘价算。
    mask0 = df_stock_d["vol"]==0
    df_stock_d.loc[mask0,"avg"]=df_stock_d.loc[mask0,"close"]
    # print(df_stock_d.loc[df_stock_d["avg"].isnull()])

    fq_cols = ["open", "high", "low", "close","avg","vol"]

    # 保存原始数据到新的列。
    for col in fq_cols:
        df_stock_d[col + "0"] = df_stock_d[col].values

    # 后复权
    fq_factor = df_stock_d["adj_factor"]
    # if qfq_type=="hfq":
    #     fq_factor = df_stock_d["adj_factor"]

    fq_factor = fq_factor.values.reshape(-1, 1).dot(np.ones((1, len(fq_cols))))

    # Deal with open,high,low,close.
    df_stock_d.loc[:, fq_cols[:5]] = df_stock_d[fq_cols[:5]] * fq_factor[:, :5]
    # Deal with vol.
    df_stock_d.loc[:, fq_cols[5]] = df_stock_d[fq_cols[5]] / fq_factor[:, 0]

    return df_stock_d


def FE_single_stock_d(df:pd.DataFrame, targets,start=None,end=None):
    df = df.sort_index(ascending=False)

    # Parameter setting
    cols_move = ["open", "high", "low", "close","avg", "vol", "amt"]
    cols_roll = ["open", "high", "low", "close","avg", "vol", "amt"]
    cols_k_line = ["open", "high", "low", "close","avg", "vol", "amt"]
    cols_fq = ["open", "high", "low", "close","avg"]
    cols_candle_stick = cols_fq

    move_upper_bound = 6
    mv_list = np.arange(0, move_upper_bound)

    candle_stick_mv_list = np.arange(0, move_upper_bound)

    kma_k_list = [3, 5, 10, 20, 60, 120, 250]
    kma_mv_list = np.arange(0, move_upper_bound)

    k_line_k_list = kma_k_list
    k_line_mv_list = np.arange(0, move_upper_bound)

    rolling_k_list = np.array(kma_k_list, dtype=int) * -1

    # Feature engineering
    df_tomorrow = FE.move(-1, df, ["open", "high", "low", "close"])

    df_qfq = df[cols_fq] / df["adj_factor"].iloc[0]
    df_qfq.columns = ["qfq_" + col for col in cols_fq]
    df_qfq["qfq_vol"]=df["vol"]*df["adj_factor"].iloc[0]
    df_tomorrow_qfq = FE.move(-1, df_qfq)

    df_targets_list = []
    for t in targets:
        pred_period = t["period"]
        if t["func"] == "min":
            df_target = FE.rolling(t["func"], pred_period, FE.move(-1, df, cols=t["col"]))
        elif t["func"] == "max":
            # df_target = rolling(t["func"],pred_period - 1, move(-2, df, cols=t["col"]))
            df_target = FE.rolling(t["func"], pred_period, FE.move(-1, df, cols=t["col"]))
        elif t["func"] == "mean":
            # df_target = rolling(t["func"], pred_period - 1, move(-2, df, cols=t["col"]))
            df_target = FE.rolling(t["func"], pred_period, FE.move(-1, df, cols=t["col"]))

            # p1 = (pred_period - 1) // 3
            # p2 = p1
            # p3 = pred_period - 1 - p1 - p2
            # df_period_mean1 = rolling(t["func"], p1, move(-2, df, t["col"]))
            # df_period_mean2 = rolling(t["func"], p2, move(-2 - p1, df, t["col"]))
            # df_period_mean3 = rolling(t["func"], p3, move(-2 - p1 - p2, df, t["col"]))
            # df_targets_list.extend([df_period_mean1,df_period_mean2,df_period_mean3])
        elif t["func"] == "avg":
            tmp = FE.rolling("sum", pred_period,
                                FE.move(-1, df, cols=["vol","amt"],prefix=False),
                                prefix=False)
            # print(df_target)
            df_target = pd.DataFrame(tmp["amt"]/tmp["vol"]*10,
                                     columns=["f{}avg_f1mv".format(
                                         pred_period)])
            # df_target.loc[tmp.index[tmp["vol"] == 0], "f{}avg_f1mv".format(pred_period)] \
            #     = df_tomorrow.loc[tmp.index[tmp["vol"] == 0], "close"]
            # print(df["code"].iloc[0])
            # print(pd.concat([df[["avg","open","close"]],df_target],
            #                 axis=1).round(2)[20:])
        else:
            raise ValueError("Fun type {} is not supported!".format(t["func"]))
        df_targets_list.append(df_target)

    df_basic_con_chg = FE.chg_rate(FE.move(1, df[cols_move]), df[cols_move])
    df_basic_mv_con_chg_list = [FE.move(i, df_basic_con_chg) for i in mv_list]
    # df_basic_mv_list = [move(i, df, cols_move) for i in mv_list]
    df_basic_mv_cur_chg_list = [FE.chg_rate(FE.move(i, df[cols_move]), df[cols_move])
                                for i in mv_list[2:]]
    df_basic_candle_stick = FE.candle_stick(df[cols_fq])
    df_basic_mv_candle_list = [FE.move(i, df_basic_candle_stick)
                           for i in mv_list]


    # df_1ma = k_MA(1, df[["vol", "amt"]])
    df_kma_list = [FE.k_MA(k, df[["vol", "amt","close"]]) for k in kma_k_list]
    df_kma_tot = pd.concat(df_kma_list,axis=1)
    df_kma_con_chg = FE.chg_rate(FE.move(1, df_kma_tot), df_kma_tot)
    df_kma_mv_con_chg_list = [FE.move(i,df_kma_con_chg) for i in mv_list]
    df_kma_mv_cur_chg_list = [FE.chg_rate(FE.move(i, df_kma_tot), df_kma_tot) for i in mv_list[2:]]
    df_kma_list = [df[["avg"]]]+df_kma_list
    df_kma_con_k_list = [FE.chg_rate(df_kma_list[i + 1], df_kma_list[i]) for i in range(len(df_kma_list) - 1)]

    # df_kma_cur_chg_list = [change_rate(k_MA(k, df[["vol", "amt"]]), df["avg"])
    #                        for k in kma_k_list]
    # df_move_kma_change_list = [move(mv, df_kma_change)
    #                            for df_kma_change in df_kma_cur_chg_list
    #                            for mv in kma_mv_list]

    df_k_line_list = [FE.k_line(k, df[cols_k_line]) for k in k_line_k_list]
    # df_k_line_tot = pd.concat(df_k_line_list,axis=1)
    df_k_line_mv_con_chg_list = [FE.move(k * mv, FE.chg_rate(FE.move(k * 1, df_k_line), df_k_line))
                                 for k,df_k_line in zip(k_line_k_list,df_k_line_list)
                                 for mv in mv_list]
    # df_k_line_mv_con_chg_list = [move(i,df_k_line_con_chg) for i in mv_list]
    df_k_line_mv_cur_chg_list = [FE.chg_rate(FE.move(k * mv, df_k_line), df_k_line)
                                 for k,df_k_line in zip(k_line_k_list,df_k_line_list)
                                 for mv in mv_list[2:]]
    # [change_rate(move(i,df_k_line_tot),df_k_line_tot) for i in mv_list[2:]]
    df_k_line_list = [df[cols_k_line]] + df_k_line_list
    df_k_line_con_k_list = [FE.chg_rate(df_k_line_list[i + 1], df_k_line_list[i]) for i in range(len(df_k_line_list) - 1)]
    # df_k_line_candle_stick = pd.concat([candle_stick(df_k_line[df_k_line.columns[:5]]) for df_k_line in df_k_line_list],axis=1)
    df_k_line_mv_candle_stick = [FE.move(k*mv,FE.candle_stick(df_k_line[df_k_line.columns[:5]]))
                                 for k,df_k_line in zip(k_line_k_list,
                                                        df_k_line_list[1:])
                                 for mv in mv_list]

    # [move(i,df_k_line_candle_stick) for i in mv_list]
    # df_change_move_k_line_list = [change_rate(move(k * mv, df_k_line),
    #                                           df[cols_k_line])
    #                               for k, df_k_line in df_k_line_list
    #                               for mv in k_line_mv_list]

    df_rolling_change_list = [
        FE.chg_rate(FE.rolling(rolling_type, days=days, df=df, cols=cols_roll),
                    df[cols_roll])
        for days in rolling_k_list
        for rolling_type in ["max", "min", "mean"]]

    df_not_in_X = pd.concat(
        [df_qfq, df_tomorrow, df_tomorrow_qfq] + df_targets_list, axis=1, sort=False)

    # df_stck = pd.concat(
    #     [df] + df_basic_mv_cur_chg_list
    #     # + df_basic_mv_candle_list
    #     # + df_move_kma_change_list
    #     + df_rolling_change_list
    #     # + df_change_move_k_line_list
    #     + [df_not_in_X],
    #     axis=1,
    #     sort=False)

    df_stck = pd.concat([df]
                        + df_basic_mv_cur_chg_list
                        + df_basic_mv_con_chg_list
                        + df_basic_mv_candle_list
                        + df_kma_mv_con_chg_list
                        + df_kma_mv_cur_chg_list
                        + df_kma_con_k_list
                        + df_k_line_mv_con_chg_list
                        + df_k_line_mv_cur_chg_list
                        + df_k_line_con_k_list
                        + df_k_line_mv_candle_stick
                        + df_rolling_change_list
                        + [df_not_in_X], axis=1, sort=False)

    # cols_not_in_X = ["code"]+list(df_not_in_X.columns)
    cols_not_in_X = list(df_not_in_X.columns)

    df_stck = df_stck.loc[IDX[start:end, :], :]
    # if start:
    #     if type(start) == str:
    #         start = int(start.replace("-",""))
    #     df_stck = df_stck.loc[IDX[start:end,:],:]
    # if end:
    #     if type(end) == str:
    #         end = int(end.replace("-",""))
    #     df_stck = df_stck[df_stck.index < end]

    # print("Stock_d columns:",len(df_stck.columns),len(cols_not_in_X))
    return df_stck, cols_not_in_X


def FE_stock_d(df_stock_d:pd.DataFrame, stock_pool=None, targets=None, start=None):
    """
    Do feature engineering on stock_daily data.

    :param df_stock_d: Stock daily data as a dataframe.
    :param stock_pool: The stock pool. If none, all stocks will be included.
    :param targets: Parameters as a [dict], describing target variables or labels.
    :param start: Lower bound of samples' dates, i.e. the start date of dataset.
    :return: A dataframe with shape (n_samples, n_features)
    """
    df_stock_d = prepare_stock_d(df_stock_d)
    df_stck_list = []
    cols_not_in_X = None

    start_time = time.time()
    i=-1
    for i,(code, df) in enumerate(df_stock_d.groupby("code")):
        if stock_pool and code not in stock_pool:
            continue

        # Initialize df.
        df = prepare_each_stock(df)

        df_stck, cols_not_in_X = FE_single_stock_d(df, targets=targets, start=start)
        df_stck_list.append(df_stck)

        if i%10==0:
            print("Finish processing {0} stocks in {1:.2f}s.".format(i, time.time() - start_time))

    df_stock_d_FE = pd.concat(df_stck_list, sort=False)
    print("Total processing time for {0} stocks:{1:.2f}s".format(i + 1, time.time() - start_time))
    print("Shape of df_stock_d_FE:", df_stock_d_FE.shape)

    # df_stock_d_FE.index.name = "date"

    return df_stock_d_FE, cols_not_in_X


def FE_stock_d_mp(df_stock_d:pd.DataFrame, stock_pool=None, targets=None, start=None,end=None):
    """
    A variant of FE_stock_d using multiprocessing.Pool.
    Experiment shows that speed increases nearly num_process times
    due to feature engineering parallelly on each stock.
    Joining(appending) operations are done serially in main process.

    :param df_stock_d:
    :param stock_pool:
    :param targets:
    :param start:
    :return:
    """
    df_stock_d = prepare_stock_d(df_stock_d)
    cols_not_in_X = None

    num_p = mp.cpu_count()
    p_pool = mp.Pool(processes=mp.cpu_count())
    q_res = queue.Queue()
    start_time = time.time()
    count_in = 0
    count_out = 0
    # df_stock_d_FE = pd.DataFrame()  # Initialize as an empty dataframe.
    df_stock_d_list = []
    for code, df in df_stock_d.groupby(level="code"):
        if stock_pool and code not in stock_pool:
            # print("skip")
            continue

        # print("stocks:", len(stock_pool))

        # Initialize df.
        df = prepare_each_stock(df)

        q_res.put(p_pool.apply_async(func=FE_single_stock_d, args=(df, targets, start,end)))
        count_in+=1

        if count_in>=num_p:
            res = q_res.get()
            df_single_stock_d_FE,cols_not_in_X = res.get()
            # df_stock_d_FE = df_stock_d_FE.append(df_single_stock_d_FE)
            df_stock_d_list.append(df_single_stock_d_FE)
            q_res.task_done()
            count_out += 1

        if count_out%10==0 and count_out>0:
            print("Finish processing {0} stocks in {1:.2f}s.".format(count_out, time.time() - start_time))

        # if count_in>=10:
        #     break

    while not q_res.empty():
        res = q_res.get()
        df_single_stock_d_FE, cols_not_in_X = res.get()
        # df_stock_d_FE = df_stock_d_FE.append(df_single_stock_d_FE)
        df_stock_d_list.append(df_single_stock_d_FE)
        q_res.task_done()
        count_out += 1
    del q_res

    df_stock_d_FE = pd.concat(df_stock_d_list, sort=False)
    print("Total processing time for {0} stocks:{1:.2f}s".format(count_out, time.time() - start_time))
    print("in",count_in,"out",count_out)
    print("Shape of df_stock_d_FE:",df_stock_d_FE.shape)
    # df_stock_d_FE.index.name = "date"

    return df_stock_d_FE, cols_not_in_X



def FE_index_d(df_idx_d: pd.DataFrame, start=None):
    index_pool = collect.get_index_pool()

    df_idx_d = prepare_index_d(df_idx_d)
    cols_move = ["open", "high", "low", "close", "vol"]
    cols_fq = ["open", "high", "low", "close","avg"]
    cols_roll = cols_move

    move_upper_bound = 6
    mv_list = np.arange(0, move_upper_bound)
    rolling_k_list = [-3, -5, -10, -20, -60, -120, -250, -500]

    df_idx_list = []
    for code, df in df_idx_d.groupby("code"):
        df = df.sort_index(ascending=False)
        del df["code"]

        # df["avg"] = (df["open"]+df["high"]+df["low"]+df["close"])/4

        # print("df",df.index.code)

        df_basic_con_chg = FE.chg_rate(FE.move(1, df[cols_move]), df[cols_move])
        df_basic_mv_con_chg_list = [FE.move(i, df_basic_con_chg) for i in mv_list]
        df_basic_mv_cur_chg_list = [
            FE.chg_rate(FE.move(i, df[cols_move]), df[cols_move]) for i in
            mv_list[2:]]
        df_basic_candle_stick = FE.candle_stick(df[cols_fq])
        df_basic_mv_candle_list = [FE.move(i, df_basic_candle_stick) for i in
                                   mv_list]
        # df_move_list = [
        #     change_rate(move(i, df, cols_move),df[cols_move]) for i in
        #     range(1, 6)]

        df_rolling_list = [
            (FE.chg_rate(FE.rolling("max", days, df, ["high", "vol"]),
                         df[["high", "vol"]], ),
             FE.chg_rate(FE.rolling("min", days, df, ["low", "vol"]),
                         df[["low", "vol"]], ),
             FE.chg_rate(FE.rolling("mean", days, df, ["open", "close", "vol"]),
                         df[["open", "close", "vol"]], ))
            for days in rolling_k_list
        ]

        df_rolling_flat_list = []
        for df_rolling_group in df_rolling_list:
            df_rolling_flat_list.extend(df_rolling_group)

        tmp_list = [df] \
                   + df_basic_mv_con_chg_list \
                   + df_basic_mv_cur_chg_list\
                   + df_basic_mv_candle_list\
                   + df_rolling_flat_list
        tmp = pd.concat(tmp_list, axis=1, sort=False)
        # print("tmp_list",[t.index.code for t in tmp_list],pd.concat(
        #     tmp_list,axis=1,sort=False).index.code)
        # print("tmp",tmp.index.code)
        name = index_pool[index_pool["code"]==code]["shortcut"].iloc[0]
        df_idx_list.append(FE._prefix((name, tmp[tmp.index>=start])))

    # print("df_idx_list:",df_idx_list[0].index.code)

    df_idx_d = pd.concat(df_idx_list, axis=1, sort=False)
    # print("df_idx_d:",df_idx_d.index.code)
    # df_idx_d.index.name="date"
    # print("Idx_d columns:",len(df_idx_d.columns))
    return df_idx_d


def prepare_stock_basic(df_stock_basic:pd.DataFrame):
    df_stock_basic = df_stock_basic.set_index("code")
    cols_category = ["area", "industry", "market", "exchange", "is_hs"]
    df_stock_basic = df_stock_basic.copy()
    # print(df_stock_basic[df_stock_basic[cols_category].isna().any(axis=1)][
    #           cols_category])
    df_stock_basic.loc[:,cols_category] = df_stock_basic[cols_category].fillna("")

    enc = preproc.OrdinalEncoder()
    val_enc = enc.fit_transform(df_stock_basic[cols_category])
    df_stock_basic.loc[:, cols_category] = val_enc
    print("basic:",np.max(val_enc))
    print(val_enc[np.isnan(val_enc)])

    if np.max(val_enc)< 2**8:
        df_stock_basic[cols_category] = df_stock_basic[cols_category].astype("uint8")
    elif np.max(val_enc) < 2**16:
        df_stock_basic[cols_category] = df_stock_basic[cols_category].astype("uint16")
    elif np.max(val_enc) < 2**32:
        df_stock_basic[cols_category] = df_stock_basic[cols_category].astype("uint32")
    elif np.max(val_enc) < 2**64:
        df_stock_basic[cols_category] = df_stock_basic[cols_category].astype("uint64")

    print(df_stock_basic.dtypes)

    return df_stock_basic,cols_category,enc


def prepare_data(cursor, targets=None, start=None, lowerbound=None, end=None,upper_bound=None,stock_pool=None):
    print("start:",start,"\tlowerbound:", lowerbound)

    # if type(lowerbound)==int:
    #     lowerbound = str(lowerbound)
    #     lowerbound = lowerbound[:4]+"-"+lowerbound[4:6]+"-"+lowerbound[6:8]
    #
    # if type(upper_bound)==int:
    #     upper_bound = str(upper_bound)
    #     upper_bound = upper_bound[:4]+"-"+upper_bound[4:6]+"-"+upper_bound[6:8]

    # Prepare df_stock_basic
    df_stock_basic = dbop.create_df(cursor, const.STOCK_BASIC[const.TABLE])
    df_stock_basic, cols_category, enc = prepare_stock_basic(df_stock_basic)
    print(df_stock_basic[cols_category].iloc[:10])

    # Prepare df_stock_d_FE
    if end:
        where_clause = "date<{0}".format(upper_bound)
        df_stock_d = dbop.create_df(cursor, const.STOCK_DAY[const.TABLE], lowerbound,where_clause=where_clause)
    else:
        df_stock_d = dbop.create_df(cursor, const.STOCK_DAY[const.TABLE], lowerbound)
    print("min_date:", min(df_stock_d.date))
    df_stock_d_FE, cols_not_in_X = FE_stock_d_mp(df_stock_d,
                                               stock_pool=stock_pool,
                                               targets=targets,
                                               start=start,end=end)

    # df_stock_d_FE = df_stock_d_FE.loc[IDX[start:end,:],:]
    print(df_stock_d_FE.shape)
    # print(df_stock_d_FE.index.name)

    # Prepare df_index_d_FE
    df_index_d = dbop.create_df(cursor, const.INDEX_DAY[const.TABLE], lowerbound)
    df_index_d_FE = FE_index_d(df_index_d, start=start)
    df_index_d_FE = df_index_d_FE.sort_index().loc[start:end,:]
    print(df_index_d_FE.shape, len(df_index_d_FE.index.unique()))
    # print(df_index_d_FE.index.name)

    print("step0")
    # Merge three df.
    df_all = df_stock_d_FE.join(df_index_d_FE, how="left")
    # df_all.index.name = "date"
    df_all = df_all.join(df_stock_basic, how="left")

    print("step1")
    df_all["list_days"] = -1
    # col_index = list(df_all.columns).index("list_days")

    #todo: change date data in stock basic to int type.
    df_all["list_date"] = df_all["list_date"].apply(lambda x:x.replace("-","")).astype(int)
    df_all["list_days"] = df_all.index.get_level_values("date")-df_all["list_date"]
    # for i in range(len(df_all.index)):
    #     date = datetime.datetime.strptime(df_all.index[i],"%Y-%m-%d")
    #     row = df_all.iloc[i]
    #     list_date = datetime.datetime.strptime(row["list_date"],"%Y%m%d")
    #     delta = date - list_date
    #     df_all.iloc[i,col_index] = delta.days
    #     if i%100==0:
    #         print("Process list_days:",i)
    print(df_all[["list_date","list_days"]].iloc[50:100])

    print("step2")
    cols_not_in_X += list(df_stock_basic.columns.difference(
        cols_category+["code"]))
    print(df_all.shape)
    print(df_all[df_all["list_days"]<0][["list_date","list_days"]])

    return df_all[df_all.columns.difference(cols_not_in_X)], df_all[cols_not_in_X], cols_category, enc


def feature_select(X, y):
    import sklearn.ensemble as ensemble
    clf = ensemble.ExtraTreesClassifier(random_state=0)
    clf.fit(X, y)
    import sklearn.feature_selection as fselect
    model = fselect.SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print("selected feature number:", X_new.shape)

    return X_new, model


def mp_stock(df_input:pd.DataFrame, target: callable, stock_pool=None, print_freq=10,**kwargs):
    num_p = mp.cpu_count()
    p_pool = mp.Pool(processes=mp.cpu_count())
    q_res = queue.Queue()
    start_time = time.time()
    count_in = 0
    count_out = 0
    df_result_list = []
    for code, df in df_input.groupby("code"):
        if stock_pool and code not in stock_pool:
            continue

        q_res.put(p_pool.apply_async(func=target, args=(df,), kwds=kwargs))
        count_in+=1

        if count_in>=num_p:
            res = q_res.get()
            df_result_list.append(res.get())
            q_res.task_done()
            count_out += 1

        if count_out % print_freq==0 and count_out>0:
            print("{0}: Finish processing {1} stocks in {2:.2f}s.".format(target.__name__,count_out, time.time() - start_time))

    while not q_res.empty():
        res = q_res.get()
        df_result_list.append(res.get())
        q_res.task_done()
        count_out += 1
    del q_res

    df_result = pd.concat(df_result_list, sort=False,axis=0)
    print("{0}: Total processing time for {0} stocks:{1:.2f}s".format(target.__name__,count_out, time.time() - start_time))
    print("in",count_in,"out",count_out)
    print("Shape of resulting dataframe:",df_result.shape)
    # df_result.index.name = "date"

    return df_result


if __name__ == '__main__':
    import db_operations as dbop
    cursor = dbop.connect_db("sqlite3").cursor()
    # targets = [{"period": 20, "func": "max", "col": "high"},
    #                       {"period": 20, "func": "min", "col": "low"},
    #                       {"period": 20, "func": "avg", "col": ""},
    #                       {"period": 5, "func": "max", "col": "high"},
    #                       {"period": 5, "func": "min", "col": "low"},
    #                       {"period": 5, "func": "avg", "col": ""},
    #                       ]
    # df1,df2,_,_ = prepare_data(cursor=cursor,targets=targets,start=20180901,lowerbound=20180101,stock_pool=["600352.SH","000581.SZ","002440.SZ"])
    # print(df1)
    from constants import *
    start = 20000101
    df = dbop.create_df(cursor, INDEX_DAY[TABLE], start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600352.SH'",
                        )
    df = prepare_index_d(df)
    print(df.head())
