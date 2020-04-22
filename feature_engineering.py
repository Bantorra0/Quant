import numpy as np
import pandas as pd
import multiprocessing as mp
import queue
import time

np.seterr(divide='ignore', invalid='ignore')
IDX = pd.IndexSlice


def _check_int(arg):
    if type(arg) not in [int,np.int,np.int8,np.int16,np.int32]:
        raise ValueError("{} is not a int".format(arg))


def _check_iterable(arg):
    if not hasattr(arg, "__iter__"):
        raise ValueError("{} is not iterable".format(arg))


def _make_iterable(arg):
    if type(arg) == str or not hasattr(arg, "__iter__"):
        return [arg]
    else:
        return arg


def _prefix(prefix, df: pd.DataFrame, copy=False):
    if copy:
        df = df.copy()
    df.columns = list(map(lambda col: str(prefix) + "_" + col, df.columns))
    return df


def move(days, df: pd.DataFrame, cols=None, prefix=True):
    _check_int(days)
    if cols is None:
        cols = df.columns
    cols = _make_iterable(cols)

    if days==0:
        return df[cols].copy()

    df = df.sort_index(ascending=False)
    if days > 0:
        pre = "p{}mv".format(abs(days))
        df_mv = df[cols].iloc[days:].copy()
        df_mv.index = df.index[:-days]
    else:
        pre = "f{}mv".format(abs(days))
        df_mv = df[cols].iloc[:days].copy()
        df_mv.index = df.index[-days:]

    if prefix:
        return _prefix(pre, df_mv)
    else:
        return df_mv


def add_int_idx(df:pd.DataFrame):
    df_single_stock_d_list = []
    for code, df_single_stock_d in df.groupby("code"):
        k = len(df_single_stock_d)
        df_single_stock_d = df_single_stock_d.sort_index(ascending=True)
        df_single_stock_d["idx"] = np.arange(k)
        df_single_stock_d_list.append(df_single_stock_d)

    df_stock_d = pd.concat(df_single_stock_d_list, axis=0)
    origin_index = df_stock_d.index
    df_stock_d.reset_index(level="code", inplace=True)
    df_stock_d.set_index(["code", "idx"], inplace=True)

    return df_stock_d,origin_index


def move_batch(days, df: pd.DataFrame, prefix=True, sort=True):
    _check_int(days)

    if days==0:
        return df.copy()
    elif days > 0:
        pre = "p{}mv".format(abs(days))
    else:
        pre = "f{}mv".format(abs(days))

    df = df.groupby(level="code").shift(days)

    if prefix:
        df = _prefix(pre,df,copy=False)
    return df


def rolling(rolling_type, days, df: pd.DataFrame, cols=None,
            prefix=True):
    """
    A wrapper of df.rolling. Current date is included when using arg days, e.g.
    days=-5 means the window is current date and previous 4 days, days=5
    means currate date and future 4 days.

    :param rolling_type:
    :param days:
    :param df:
    :param cols:
    :param prefix:
    :return:
    """
    _check_int(days)
    if cols is None:
        cols = df.columns
    cols = _make_iterable(cols)

    df = df.sort_index(ascending=False)
    period = abs(days)
    if rolling_type == "max":
        df_rolling = df[cols].rolling(window=abs(days)).max()
    elif rolling_type == "min":
        df_rolling = df[cols].rolling(window=abs(days)).min()
    elif rolling_type == "mean":
        df_rolling = df[cols].rolling(window=abs(days)).mean()
    elif rolling_type == "sum":
        df_rolling = df[cols].rolling(window=abs(days)).sum()
    elif rolling_type == "std":
        df_rolling = df[cols].rolling(window=abs(days)).std()
    else:
        raise ValueError(
            "rolling_type='{}' is not supported.".format(rolling_type))

    n = len(df_rolling)
    idxes = df_rolling.index
    if days > 0:
        pre = "f" + str(abs(days)) + rolling_type
        df_rolling = df_rolling.iloc[period - 1:n]
        df_rolling.index = idxes[period - 1:n]
    else:
        pre = "p" + str(abs(days)) + rolling_type
        df_rolling = df_rolling.iloc[period - 1:n]
        if n - period + 1 >= 0:
            df_rolling.index = idxes[:n - period + 1]

    if prefix:
        return _prefix(pre, df_rolling)
    else:
        return df_rolling


def rolling_batch(ops, days, df: pd.DataFrame,
                  prefix=True, sort=False):
    """
    A wrapper of df.rolling. Current date is included when using arg days, e.g.
    days=-5 means the window is current date and previous 4 days, days=5
    means currate date and future 4 days.

    :param ops:
    :param days:
    :param df:
    :param cols:
    :param prefix:
    :return:
    """
    _check_int(days)
    ops = _make_iterable(ops)

    if sort:
        df = df.sort_index(ascending=True)

    if days>0:
        pre = "p{}".format(abs(days))
    elif days<0:
        df = df.sort_index(ascending=False)
        pre = "f{}".format(abs(days))
    else:
        raise ValueError("days==0 is illegal!")
    window = abs(days)

    results = {op:[] for op in ops}
    for _, group in df.groupby(level="code"):
        for op in ops:
            results[op].append(group.rolling(window=window).agg(op))

    if prefix:
        result = pd.concat(
            [pd.concat(results[op],axis=0).rename(columns={col:pre+op+"_"+col for col in df.columns}) for op in ops],
            axis=1)
    else:
        result = pd.concat(
            [pd.concat(results[op], axis=0) for op in ops],axis=1)
    return result


def rolling_batch2(ops, days, df: pd.DataFrame,
                  prefix=True, sort=False):
    """
    A wrapper of df.rolling. Current date is included when using arg days, e.g.
    days=-5 means the window is current date and previous 4 days, days=5
    means current date and future 4 days.

    :param ops:
    :param days:
    :param df:
    :param cols:
    :param prefix:
    :return:
    """
    _check_int(days)
    ops = _make_iterable(ops)

    if sort:
        df = df.sort_index(ascending=True)

    if days>0:
        pre = "p{}".format(abs(days))
    elif days<0:
        df = df.sort_index(ascending=False)
        pre = "f{}".format(abs(days))
    else:
        raise ValueError("days==0 is illegal!")
    window = abs(days)

    # results=[group.rolling(window=window).agg(ops)
    #          for _, group in df.groupby(level="code")]
    # columns = results[0].columns
    # result = pd.concat(results,axis=0)
    result = groupby_rolling(df,level="code",window=window,ops=ops,check_col="open")
    columns = df.columns
    # cols1,cols2 = columns.get_level_values(0),\
    #               columns.get_level_values(1)

    if prefix:
        # result.columns = [pre+s2+"_"+s1 for s1,s2 in zip(cols1,cols2)]
        if type(ops) == list and type(ops[0]) != tuple:
            result.columns = [pre+op+"_"+col for op in ops for col in columns]
        elif type(ops) == list and type(ops[0]) == tuple:
            result.columns = [pre+op+"_"+col for op,col in zip(ops,columns)]
    else:
        # result.columns = [s1 for s1,_ in zip(cols1,cols2)]
        if type(ops) == list and type(ops[0]) != tuple:
            result.columns = [col for op in ops for col in columns ]
        elif type(ops) == list and type(ops[0]) == tuple:
            result.columns = [col for op,col in zip(ops,columns)]

    # print(result.columns)
    return result


def groupby_rolling2(df:pd.DataFrame,by=None,level=None,window=None,
                     ops=None,check_col="open",sort=False):
    if sort:
        df=df.sort_index()

    result = df.rolling(window).agg(ops)
    result.loc[df[check_col].groupby(by=by,level=level).shift(window-1).isnull()]=np.nan
    return result


def groupby_rolling(df:pd.DataFrame,by=None,level=None,window=None,
                     ops=None,check_col="open",sort=False):
    if sort:
        df=df.sort_index()

    if type(ops)==list and type(ops[0])!=tuple:
        result = pd.concat([df.rolling(window).agg(op) for op in ops],
                           axis=1,sort=False)
    elif type(ops)==list and type(ops[0])==tuple:
        result = pd.concat(
            [df[col].rolling(window).agg(op) for col,op in ops],
            axis=1,sort=False)
    else:
        result = df.rolling(window).agg(ops)
    cond = df[[col for col in df.columns if col in {check_col, by}]].groupby(by=by, level=level).shift(window - 1)\
        .isnull().any(axis=1).values
    result.loc[cond]=np.nan
    return result


def chg_rate(df1: pd.DataFrame, df2: pd.DataFrame, cols1=None,
             cols2=None, prefix=True):
    if cols1:
        df1 = df1[cols1].copy()
    else:
        df1 = df1.copy()

    if cols2:
        df2 = df2[cols2].copy()
    else:
        df2 = df2.copy()

    if df1.shape[1] != df2.shape[1]:
        raise ValueError(
            "Column length not the same:{0}!={1}".format(df1.shape[1],
                                                         df2.shape[1]))

    cols1 = df1.columns
    cols2 = df2.columns
    # Use df rather than np.array because data calculation need to be aligned with df.index(date),
    # especially when len(df1.index)!=len(df2.index)
    # Reset columns to make sure columns of df1 and df2 are the same,
    # because operations are based on index and columns.
    df2.columns = cols1
    df3 = df2/df1-1
    df3 = df3*100  # Convert to percentage.
    cols = ["({1}/{0}-1)".format(c1, c2) for c1, c2 in zip(cols1, cols2)]
    df3.columns = cols
    # Round to two decimals and convert to float16 to save memory.
    return df3.astype('float16')


def chg_rate_batch(df1: pd.DataFrame, df2: pd.DataFrame, sort=True):
    if df1.shape[1] != df2.shape[1]:
        raise ValueError(
            "Column length not the same:{0}!={1}".format(df1.shape[1],
                                                         df2.shape[1]))
    if sort:
        df1 = df1.sort_index()
        df2 = df2.sort_index()

    if not (df1.index==df2.index).all().all():
        raise ValueError("df1.index is not the same as df2.index!")

    cols = ["({1}/{0}-1)".format(c1, c2) for c1, c2 in zip(df1.columns, df2.columns)]
    result = pd.DataFrame(index=df1.index,columns=cols)
    result.iloc[:] = (df2.values/df1.values-1)*100
    # Round to two decimals and convert to float16 to save memory.
    return result.astype('float16')
    # return result

def candle_stick(df:pd.DataFrame):
    df_result = pd.DataFrame(index=df.index)
    # if df.shape[1]!=5:
    #     raise ValueError("df.shape[1] {}!=5".format(df.shape[1]))

    cols = ["open","high","low","close","avg"]
    open,high,low,close,avg = [col_name for col_type in cols for col_name in df.columns if col_type in col_name]
    # open,high,low,close,avg = df.columns

    # avg = (df[open]+df[high]+df[low]+df[close])/4

    stick_top = df.apply(lambda x:x[open] if x[open]>x[close] else x[close],
                         axis=1)
    stick_bottom = df.apply(lambda x: x[open] if x[open] < x[close] else x[close],
                         axis=1)

    df_result["({0}-{1})/{2}".format(high, low, avg)] = \
        (df[high] - df[low]) / df[avg]
    df_result["({0}-{1})/{2}".format(close, open, avg)] = \
        (df[close] - df[open]) / df[avg]

    df_result["({0}-{1})/{2}".format(high, open, avg)] = \
        (df[high] - df[open]) / df[avg]
    df_result["({0}-{1})/{2}".format(low, open, avg)] = \
        (df[low] - df[open]) / df[avg]

    df_result["({0}-{1})/{2}".format(high, close, avg)] = \
        (df[high] - df[close]) / df[avg]
    df_result["({0}-{1})/{2}".format(low, close, avg)] = \
        (df[low] - df[close]) / df[avg]

    df_result["upper_shadow/{0}".format(avg)] = \
        (df[high] - stick_top) / df[avg]
    df_result["lower_shadow/{0}".format(avg)] = \
        (stick_bottom - df[low]) / df[avg]

    df_result = df_result*100
    return df_result.astype('float16')


def candle_stick_batch(df:pd.DataFrame):
    df_result = pd.DataFrame(index=df.index)
    # if df.shape[1]!=5:
    #     raise ValueError("df.shape[1] {}!=5".format(df.shape[1]))

    cols = ["open","high","low","close","avg"]
    open,high,low,close,avg = [col_name for col_type in cols for col_name in df.columns if col_type in col_name]

    stick_top,stick_bottom = df[[open,close]].max(axis=1),df[[open,close]].min(axis=1)

    df_result["({0}-{1})/{2}".format(high, low, avg)] = \
        (df[high] - df[low])
    df_result["({0}-{1})/{2}".format(close, open, avg)] = \
        (df[close] - df[open])

    df_result["({0}-{1})/{2}".format(high, open, avg)] = \
        (df[high] - df[open])
    df_result["({0}-{1})/{2}".format(low, open, avg)] = \
        (df[low] - df[open])

    df_result["({0}-{1})/{2}".format(high, close, avg)] = \
        (df[high] - df[close])
    df_result["({0}-{1})/{2}".format(low, close, avg)] = \
        (df[low] - df[close])

    df_result["upper_shadow/{0}".format(avg)] = \
        (df[high] - stick_top)
    df_result["lower_shadow/{0}".format(avg)] = \
        (stick_bottom - df[low])

    df_result = df_result*100 / (df[avg].values.reshape(-1,1).dot(np.ones((1,df_result.shape[1]))))
    return df_result.astype('float16')
    # return df_result


def k_MA(k:int, df:pd.DataFrame):
    # if df.shape[1] != 2:
    #     raise ValueError("df.shape[1] {}!=2".format(df.shape[1]))

    if "amt" not in df.columns:
        raise ValueError("\"amt\" not in df.columns")
    elif "vol" not in df.columns:
        raise ValueError("\"vol\" not in df.columns")
    elif "close" not in df.columns:
        raise ValueError("\"close\" not in df.columns")

    df_result = pd.DataFrame(index=df.index)
    df = df.sort_index(ascending=True)

    df_result["{}MA_vol".format(k)] = df["vol"].rolling(window=k).mean()
    df_result["{}MA_amt".format(k)] = df["amt"].rolling(window=k).mean()
    df_result["{}MA".format(k)] = df_result["{}MA_amt".format(k)]\
                                  /df_result["{}MA_vol".format(k)]*10

    # 如果累计vol=0（计算区间内停牌），则上述计算结果为inf，取收盘价。
    # 此处假设停牌期间已完成数据填充，按vol=0，amt=0，其他价格按前一天（停牌前最后一天）收盘价算。
    df_result.loc[df_result.index[df_result["{}MA_vol".format(k)] == 0],"{}MA".format(k)] = \
        df.loc[df_result.index[df_result["{}MA_vol".format(k)] == 0],"close"]
    return df_result[["{}MA".format(k)]].sort_index(ascending=False)


def k_MA_batch(k:int, df:pd.DataFrame, sort=True):
    if "amt" not in df.columns:
        raise ValueError("\"amt\" not in df.columns")
    elif "vol" not in df.columns:
        raise ValueError("\"vol\" not in df.columns")
    elif "close" not in df.columns:
        raise ValueError("\"close\" not in df.columns")

    if sort:
        df = df.sort_index()

    ops = [("vol","sum"),("amt","sum")]
    ops = dict(ops)
    result = groupby_rolling(df,level="code",window=k,ops=ops,check_col="open")
    # result = pd.concat([group.rolling(k).sum() for _,group in df[["vol",
    #                                                                  "amt"]].groupby(level="code")])
    result.rename(columns={col:"{}MA_".format(k)+col for col in result.columns},inplace=True)

    result["{}MA".format(k)] = result["{}MA_amt".format(k)]/result["{}MA_vol".format(k)]*10

    # 如果累计vol=0（计算区间内停牌），则上述计算结果为inf，取收盘价。
    # 此处假设停牌期间已完成数据填充，按vol=0，amt=0，其他价格按前一天（停牌前最后一天）收盘价算。
    result.loc[result["{}MA_vol".format(k)] == 0,"{}MA".format(k)] = \
        df.loc[result["{}MA_vol".format(k)] == 0,"close"]
    return result[["{}MA".format(k)]]


def k_line(k:int, df:pd.DataFrame):
    # if df.shape[1] != 7:
    #     raise ValueError("df.shape[1] {}!=7".format(df.shape[1]))

    cols = ["open","high","low","close","vol","amt"]
    if not set(cols).issubset(set(df.columns)):
        raise ValueError(str(cols)+" is not a subset of {}".format(set(df.columns)))

    df = df.sort_index(ascending=True)
    df_result = pd.DataFrame(index=df.index)

    df_result["{}k_open".format(k)] = pd.Series(np.array(df["open"].iloc[:-k + 1]),index=df.index[k - 1:])
    df_result["{}k_high".format(k)] = df["high"].rolling(k).max()
    df_result["{}k_low".format(k)] = df["low"].rolling(k).min()
    df_result["{}k_close".format(k)]=df["close"]
    df_result["{}k_vol".format(k)] = df["vol"].rolling(k).mean()
    df_result["{}k_amt".format(k)] = df["amt"].rolling(k).mean()
    df_result["{}k_avg".format(k)] = df_result["{}k_amt".format(k)]/df_result["{}k_vol".format(k)] * 10

    df_result.loc[df_result.index[df_result["{}k_vol".format(k)]==0],
                  "{}k_avg".format(k)] = \
        df_result.loc[df_result.index[df_result["{}k_vol".format(k)]==0],
                      "{}k_close".format(k)]
    output_cols = ["open","high","low","close","avg","vol","amt"]
    # output_cols = ["open","high","low","close","vol","amt","avg"]

    return df_result[["{0}k_{1}".format(k,col) for col in output_cols]]


def k_line_batch(k:int, df:pd.DataFrame,sort=True):
    cols = ["open","high","low","close","vol","amt"]
    if not set(cols).issubset(set(df.columns)):
        raise ValueError(str(cols)+" is not a subset of {}".format(set(df.columns)))

    if sort:
        df = df.sort_index()

    # results = {col:[] for col in cols if col!="close"}
    results = []
    for _, group in df[cols].groupby(level="code"):
        # results["open"].append(group["open"].shift(k-1))
        # results["high"].append(group["high"].rolling(k).max())
        # results["low"].append(group["low"].rolling(k).min())
        # results["vol"].append(group["vol"].rolling(k).mean())
        # results["amt"].append(group["amt"].rolling(k).mean())
        results.append(group.rolling(k).agg({"high":"max","low":"min","vol":"mean","amt":"mean"}))

    ops = [("high", "max"), ("low", "min"), ("vol", "mean"), ("amt", "mean")]
    ops = dict(ops)
    result = groupby_rolling(df,level="code",window=k,ops=ops,check_col="open")

    # result = pd.concat(results,axis=0)
    result["open"] =df["open"].groupby(level="code").shift(k-1)

    result["close"] = df["close"]
    result["avg"] = result["amt"]/result["vol"]*10
    result.loc[result["vol"]==0,"avg"] = result.loc[result["vol"]==0,"close"]

    output_cols = ["open","high","low","close","avg","vol","amt"]
    # output_cols = ["open","high","low","close","vol","amt","avg"]

    column_mapper = {col:"{}k_".format(k)+col for col in output_cols}
    result.rename(columns=column_mapper,inplace=True)
    return result[[column_mapper[col] for col in output_cols]]


def stock_d_FE(df:pd.DataFrame, targets,start=None,end=None):
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

    # ture engineering
    df_tomorrow = move(-1, df, ["open", "high", "low", "close"])

    df_qfq = df[cols_fq] / df["adj_factor"].iloc[0]
    df_qfq.columns = ["qfq_" + col for col in cols_fq]
    df_qfq["qfq_vol"]=df["vol"]*df["adj_factor"].iloc[0]
    df_tomorrow_qfq = move(-1, df_qfq)

    df_targets_list = []
    for t in targets:
        pred_period = t["period"]
        if t["func"] == "min":
            df_target = rolling(t["func"], pred_period, move(-1, df, cols=t["col"]))
        elif t["func"] == "max":
            # df_target = rolling(t["func"],pred_period - 1, move(-2, df, cols=t["col"]))
            df_target = rolling(t["func"], pred_period, move(-1, df, cols=t["col"]))
        elif t["func"] == "mean":
            # df_target = rolling(t["func"], pred_period - 1, move(-2, df, cols=t["col"]))
            df_target = rolling(t["func"], pred_period, move(-1, df, cols=t["col"]))

            # p1 = (pred_period - 1) // 3
            # p2 = p1
            # p3 = pred_period - 1 - p1 - p2
            # df_period_mean1 = rolling(t["func"], p1, move(-2, df, t["col"]))
            # df_period_mean2 = rolling(t["func"], p2, move(-2 - p1, df, t["col"]))
            # df_period_mean3 = rolling(t["func"], p3, move(-2 - p1 - p2, df, t["col"]))
            # df_targets_list.extend([df_period_mean1,df_period_mean2,df_period_mean3])
        elif t["func"] == "avg":
            tmp = rolling("sum", pred_period,
                                move(-1, df, cols=["vol","amt"],prefix=False),
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

    df_basic_con_chg = chg_rate(move(1, df[cols_move]), df[cols_move])
    df_basic_mv_con_chg_list = [move(i, df_basic_con_chg) for i in mv_list]
    # df_basic_mv_list = [move(i, df, cols_move) for i in mv_list]
    df_basic_mv_cur_chg_list = [chg_rate(move(i, df[cols_move]), df[cols_move])
                                for i in mv_list[2:]]
    df_basic_candle_stick = candle_stick(df[cols_fq])
    df_basic_mv_candle_list = [move(i, df_basic_candle_stick)
                           for i in mv_list]


    # df_1ma = k_MA(1, df[["vol", "amt"]])
    df_kma_list = [k_MA(k, df[["vol", "amt","close"]]) for k in kma_k_list]
    df_kma_tot = pd.concat(df_kma_list,axis=1)
    df_kma_con_chg = chg_rate(move(1, df_kma_tot), df_kma_tot)
    df_kma_mv_con_chg_list = [move(i,df_kma_con_chg) for i in mv_list]
    df_kma_mv_cur_chg_list = [chg_rate(move(i, df_kma_tot), df_kma_tot) for i in mv_list[2:]]
    df_kma_list = [df[["avg"]]]+df_kma_list
    df_kma_con_k_list = [chg_rate(df_kma_list[i + 1], df_kma_list[i]) for i in range(len(df_kma_list) - 1)]

    # df_kma_cur_chg_list = [change_rate(k_MA(k, df[["vol", "amt"]]), df["avg"])
    #                        for k in kma_k_list]
    # df_move_kma_change_list = [move(mv, df_kma_change)
    #                            for df_kma_change in df_kma_cur_chg_list
    #                            for mv in kma_mv_list]

    df_k_line_list = [k_line(k, df[cols_k_line]) for k in k_line_k_list]
    # df_k_line_tot = pd.concat(df_k_line_list,axis=1)
    df_k_line_mv_con_chg_list = [move(k * mv, chg_rate(move(k * 1, df_k_line), df_k_line))
                                 for k,df_k_line in zip(k_line_k_list,df_k_line_list)
                                 for mv in mv_list]
    # df_k_line_mv_con_chg_list = [move(i,df_k_line_con_chg) for i in mv_list]
    df_k_line_mv_cur_chg_list = [chg_rate(move(k * mv, df_k_line), df_k_line)
                                 for k,df_k_line in zip(k_line_k_list,df_k_line_list)
                                 for mv in mv_list[2:]]
    # [change_rate(move(i,df_k_line_tot),df_k_line_tot) for i in mv_list[2:]]
    df_k_line_list = [df[cols_k_line]] + df_k_line_list
    df_k_line_con_k_list = [chg_rate(df_k_line_list[i + 1], df_k_line_list[i]) for i in range(len(df_k_line_list) - 1)]
    # df_k_line_candle_stick = pd.concat([candle_stick(df_k_line[df_k_line.columns[:5]]) for df_k_line in df_k_line_list],axis=1)
    df_k_line_mv_candle_stick = [move(k*mv,candle_stick(df_k_line))
                                 for k,df_k_line in zip(k_line_k_list,
                                                        df_k_line_list[1:])
                                 for mv in mv_list]

    # [move(i,df_k_line_candle_stick) for i in mv_list]
    # df_change_move_k_line_list = [change_rate(move(k * mv, df_k_line),
    #                                           df[cols_k_line])
    #                               for k, df_k_line in df_k_line_list
    #                               for mv in k_line_mv_list]

    df_rolling_change_list = [
        chg_rate(rolling(rolling_type, days=days, df=df, cols=cols_roll),
                 df[cols_roll])
        for days in rolling_k_list
        for rolling_type in ["max", "min", "mean"]]

    df_not_in_X = pd.concat(
        [df_qfq, df_tomorrow, df_tomorrow_qfq] + df_targets_list, axis=1, sort=False)

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
                        + [df_not_in_X]
                        ,axis=1, sort=False)

    cols_not_in_X = list(df_not_in_X.columns)
    df_stck = df_stck.loc[IDX[start:end, :], :]
    return df_stck, cols_not_in_X


def stock_d_FE_batch(df:pd.DataFrame, targets,start=None,end=None,fe_list=None):
    df.sort_index(inplace=True)

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

    rolling_k_list = np.array(kma_k_list, dtype=int)

    df_fe_list = []

    cols_not_in_X=None
    if fe_list is None or fe_list["not_in_X"]:
        df_tomorrow = move_batch(-1, df[["open", "high", "low", "close"]])

        adj_factor = df["adj_factor"].groupby(level="code").transform(lambda x:np.ones(len(x))*x.iloc[-1])
        df_qfq = df[cols_fq] / (adj_factor.values.reshape(-1,1) * np.ones((1,len(cols_fq))))

        df_qfq.columns = ["qfq_" + col for col in cols_fq]
        df_qfq["qfq_vol"]=df["vol"]*adj_factor.values
        df_tomorrow_qfq = move_batch(-1, df_qfq)

        df_targets_list = []
        for t in targets:
            pred_period = t["period"]
            cols = [t["col"]]
            if t["func"] == "min":
                df_target = rolling_batch(t["func"], -pred_period, move_batch(-1, df[cols]))
            elif t["func"] == "max":
                df_target = rolling_batch(t["func"], -pred_period, move_batch(-1, df[cols]))
            elif t["func"] == "mean":
                df_target = rolling_batch(t["func"], -pred_period, move_batch(-1, df[cols]))

            elif t["func"] == "avg":
                tmp = rolling_batch("sum", -pred_period,
                                    move_batch(-1, df[["vol","amt"]],prefix=False),
                                    prefix=False)
                df_target = pd.DataFrame(tmp["amt"]/tmp["vol"]*10,
                                         columns=["f{}avg_f1mv".format(
                                             pred_period)])
            else:
                raise ValueError("Fun type {} is not supported!".format(t["func"]))
            df_targets_list.append(df_target)

        df_not_in_X = pd.concat(
            [df_qfq, df_tomorrow, df_tomorrow_qfq] + df_targets_list, axis=1, sort=False)
        df_fe_list.append(df_not_in_X)
        cols_not_in_X = list(df_not_in_X.columns)


    if fe_list is None or fe_list["basic"]:
        df_basic_con_chg = chg_rate_batch(move_batch(1, df[cols_move]), df[cols_move])
        df_basic_mv_con_chg_list = [move_batch(i, df_basic_con_chg) for i in mv_list]
        # df_basic_mv_list = [move(i, df, cols_move) for i in mv_list]
        df_basic_mv_cur_chg_list = [chg_rate_batch(move_batch(i, df[cols_move]), df[cols_move])
                                    for i in mv_list[2:]]
        df_basic_candle_stick = candle_stick_batch(df[cols_fq])
        df_basic_mv_candle_list = [move_batch(i, df_basic_candle_stick)
                               for i in mv_list]
        df_fe_list.extend([df]
                          + df_basic_mv_cur_chg_list
                          + df_basic_mv_con_chg_list
                          + df_basic_mv_candle_list)

    if fe_list is None or fe_list["kma"]:
        # df_1ma = k_MA(1, df[["vol", "amt"]])
        df_kma_list = [k_MA_batch(k, df) for k in kma_k_list]
        df_kma_tot = pd.concat(df_kma_list,axis=1,sort=False)
        df_kma_con_chg = chg_rate_batch(move_batch(1, df_kma_tot), df_kma_tot)
        df_kma_mv_con_chg_list = [move_batch(i,df_kma_con_chg) for i in mv_list]
        df_kma_mv_cur_chg_list = [chg_rate_batch(move_batch(i, df_kma_tot), df_kma_tot) for i in mv_list[2:]]
        df_kma_list = [df[["avg"]]]+df_kma_list
        df_kma_con_k_list = [chg_rate_batch(df_kma_list[i + 1], df_kma_list[i]) for i in range(len(df_kma_list) - 1)]
        df_fe_list.extend(df_kma_mv_con_chg_list
                          + df_kma_mv_cur_chg_list
                          + df_kma_con_k_list)

    if fe_list is None or fe_list["k_line"]:
        df_k_line_list = [k_line_batch(k, df[cols_k_line]) for k in k_line_k_list]
        # df_k_line_tot = pd.concat(df_k_line_list,axis=1)
        df_k_line_mv_con_chg_list = [move_batch(k * mv, chg_rate_batch(move_batch(k * 1, df_k_line), df_k_line))
                                     for k,df_k_line in zip(k_line_k_list,df_k_line_list)
                                     for mv in mv_list]
        # df_k_line_mv_con_chg_list = [move(i,df_k_line_con_chg) for i in mv_list]
        df_k_line_mv_cur_chg_list = [chg_rate_batch(move_batch(k * mv, df_k_line), df_k_line)
                                     for k,df_k_line in zip(k_line_k_list,df_k_line_list)
                                     for mv in mv_list[2:]]
        # [change_rate(move(i,df_k_line_tot),df_k_line_tot) for i in mv_list[2:]]
        df_k_line_list = [df[cols_k_line]] + df_k_line_list
        df_k_line_con_k_list = [chg_rate_batch(df_k_line_list[i + 1], df_k_line_list[i])
                                for i in range(len(df_k_line_list) - 1)]
        df_k_line_mv_candle_stick = [move_batch(k*mv,candle_stick_batch(df_k_line))
                                     for k,df_k_line in zip(k_line_k_list,df_k_line_list[1:])
                                     for mv in mv_list]
        df_fe_list.extend(df_k_line_mv_con_chg_list
                          + df_k_line_mv_cur_chg_list
                          + df_k_line_con_k_list
                          + df_k_line_mv_candle_stick)

    if fe_list is None or fe_list["rolling"]:
        ops = ["max", "min", "mean"]
        rolling_batch_cols = []
        [rolling_batch_cols.extend([col]*len(ops)) for col in cols_roll]
        # print(rolling_batch_cols)
        # df_rolling_change_list = [
        #     chg_rate_batch(rolling_batch2(ops, days=days, df=df[cols_roll]),
        #                    df[rolling_batch_cols])
        #     for days in rolling_k_list]
        df_rolling_change_list = [
            chg_rate_batch(rolling_batch2(ops, days=days, df=df[cols_roll]),
                           df[cols_roll*len(ops)]) for days in rolling_k_list]

        df_fe_list.extend(df_rolling_change_list)

    # df_not_in_X = pd.concat(
    #     [df_qfq, df_tomorrow, df_tomorrow_qfq] + df_targets_list, axis=1, sort=False)

    df_stck = pd.concat(df_fe_list
                        # +[df]
                        # + df_basic_mv_cur_chg_list
                        # + df_basic_mv_con_chg_list
                        # + df_basic_mv_candle_list
                        # + df_kma_mv_con_chg_list
                        # + df_kma_mv_cur_chg_list
                        # + df_kma_con_k_list
                        # + df_k_line_mv_con_chg_list
                        # + df_k_line_mv_cur_chg_list
                        # + df_k_line_con_k_list
                        # + df_k_line_mv_candle_stick
                        # + df_rolling_change_list
                        # + [df_not_in_X]
                        , axis=1, sort=False)

    # cols_to_round = [col for col in df_stck.columns if "/" in col]
    # print(len(cols_to_round))
    # df_stck[cols_to_round] = df_stck[cols_to_round].astype("float16")

    df_stck = df_stck.loc[IDX[start:end, :], :]
    return df_stck, cols_not_in_X


def mp_batch(df, target: callable, batch_size=10, print_freq=1, num_reserved_cpu=1,**kwargs):
    num_p = mp.cpu_count()-num_reserved_cpu
    p_pool = mp.Pool(processes=mp.cpu_count())
    q_res = queue.Queue()
    count_in,count_out = 0,0

    pool = sorted(df.index.levels[0])
    n,k = len(pool),batch_size
    groups = [df.loc[IDX[pool[i*k:(i+1)*k],:],:] for i in range(int(n/k)+1)]
    groups = [df for df in groups if len(df)>0]
    df_result_list = []
    other_result = None

    start_time = time.time()
    for gp in groups:
        print(gp.shape)
        q_res.put(p_pool.apply_async(func=target, args=(gp,), kwds=kwargs))
        count_in+=1

        if count_in>=num_p:
            res = q_res.get()
            result = res.get()
            if type(result)!=tuple:
                result = (result,)
            df_result_list.append(result[0])
            if other_result is None and len(result)>1:
                other_result = result[1:]
            q_res.task_done()
            count_out += 1

        if count_out % print_freq==0 and count_out>0:
            print("{0}: Finish processing {1} group(s) in {2:.2f}s.".format(target.__name__,count_out, time.time() - start_time))

    while not q_res.empty():
        res = q_res.get()
        result = res.get()
        if type(result) != tuple:
            result = (result,)
        df_result_list.append(result[0])
        q_res.task_done()
        count_out += 1
    del q_res

    print(type(df_result_list))
    df_result = pd.concat(df_result_list, sort=False,axis=0)
    print("{0}: Total processing time for {1} groups:{2:.2f}s".format(target.__name__,count_out, time.time() - start_time))
    print("in",count_in,"out",count_out)
    print("Shape of resulting dataframe:",df_result.shape)
    # df_result.index.name = "date"

    return df_result,other_result


def return_script(df):
    import script
    kwargs = {"loss_limit":0.05,"retracement_inc_pct":0.1,
              "max_days":20,"new_high_days_limit":8,
              "stop_profit":0.15,
              "is_truncated":False}
    df_r_spl, _ = mp_batch(df, target=script.get_return_rate_batch, batch_size=50,
                       num_reserved_cpu=1,**kwargs)
    # script.get_return_rate_batch(df,**kwargs)
    print(df_r_spl.info(memory_usage="deep"))
    df_r_spl.to_parquet(
        r"database\return_spl_{0:.0%}_{1:.0%}_{2}_{3}_{4:.0%}".format(
            kwargs["loss_limit"],kwargs["retracement_inc_pct"],
            kwargs["max_days"],kwargs["new_high_days_limit"],
            kwargs["stop_profit"] if "stop_profit" in kwargs.keys() and
                                     kwargs["stop_profit"] else float('inf'),
                            ),
        engine="pyarrow")

    df_r_spc, _ = mp_batch(df, target=script.get_return_rate_batch2,
                           batch_size=50, num_reserved_cpu=1, **kwargs)
    # script.get_return_rate_batch(df,**kwargs)
    print(df_r_spc.info(memory_usage="deep"))
    df_r_spc.to_parquet(
        r"database\return_spc_{0:.0%}_{1:.0%}_{2}_{3}_{4:.0%}".format(
            kwargs["loss_limit"], kwargs["retracement_inc_pct"],
            kwargs["max_days"], kwargs["new_high_days_limit"],
            kwargs["stop_profit"] if "stop_profit" in kwargs.keys() and kwargs[
                "stop_profit"] else float('inf'), ), engine="pyarrow")


if __name__ == '__main__':
    import db_operations as dbop
    from constants import *
    import data_process as dp
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20100101
    df = dbop.create_df(cursor, STOCK_DAY[TABLE],
                        start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600350.SH'",
                        )
    df = dp.proc_stock_d(dp.prepare_stock_d(df))
    # print(df.shape)
    # import collect
    # pool = sorted(collect.get_stock_pool())[:5]
    # df = df.loc[IDX[:,pool],:]
    print(df.shape)
    return_script(df)



    # n = 5
    # cols = ["open", "high", "low", "close","avg"]
    # # # expected = pd.concat([move(n, group) for _, group in df[cols].groupby(level="code")])\
    # # #     .dropna().sort_index()
    # # df,_ = add_int_idx(df)
    # import time
    # # df_mv = move_batch(5,df,sort=False)
    # t0 = time.time()
    # [group for _,group in df.groupby("code")]
    # print(time.time() - t0)
    # t0=time.time()
    # df_candle_stick = candle_stick(df[cols])
    # print(time.time()-t0)
    # print((df_candle_stick==df_candle_stick_batch).all().all())

    # print(time.time()-t0)
    # print((actual == expected).all().all())


