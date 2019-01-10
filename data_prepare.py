import numpy as np
import pandas as pd
import sklearn as sk

import collect as clct
import constants as const
import db_operations as dbop


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


def rolling(rolling_type, days, df: pd.DataFrame, cols=None,
            prefix=True):
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


def change_rate(df1: pd.DataFrame, df2: pd.DataFrame, cols1=None,
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
    cols = ["({1}/{0}-1)".format(c1, c2) for c1, c2 in zip(cols1, cols2)]
    df3.columns = cols
    return df3


def candle_stick(df:pd.DataFrame):
    df_result = pd.DataFrame(index=df.index)
    if df.shape[1]!=5:
        raise ValueError("df.shape[1] {}!=5".format(df.shape[1]))

    open,high,low,close,avg = df.columns

    # avg = (df[open]+df[high]+df[low]+df[close])/4

    stick_top = df.apply(lambda x:x[open] if x[open]>x[close] else x[close],
                         axis=1)
    stick_bottom = df.apply(lambda x: x[open] if x[open] < x[close] else x[close],
                         axis=1)

    df_result["(high-low)/avg"] = (df[high]-df[low])/df[avg]
    df_result["(close-open)/avg"] = (df[close] - df[open]) / df[avg]

    df_result["(high-open)/avg"] = (df[high]-df[open])/df[avg]
    df_result["(low-open)/avg"] = (df[low] - df[open]) / df[avg]

    df_result["(high-close)/avg"] = (df[high] - df[close]) / df[avg]
    df_result["(low-close)/avg"] = (df[low] - df[close]) / df[avg]

    df_result["upper_shadow/avg"] = (df[high] - stick_top) / df[avg]
    df_result["lower_shadow/avg"] = (stick_bottom - df[low]) / df[avg]

    return df_result


def k_MA(k:int, df:pd.DataFrame):
    if df.shape[1] != 2:
        raise ValueError("df.shape[1] {}!=2".format(df.shape[1]))

    if "amt" not in df.columns:
        raise ValueError("\"amt\" not in df.columns")
    elif "vol" not in df.columns:
        raise ValueError("\"vol\" not in df.columns")

    df_result = pd.DataFrame(index=df.index)
    df = df.sort_index(ascending=True)

    df_result["{}MA_vol".format(k)] = df["vol"].rolling(window=k).mean()
    df_result["{}MA_amt".format(k)] = df["amt"].rolling(window=k).mean()
    df_result["{}MA".format(k)] = df_result["{}MA_amt".format(k)]\
                                  /df_result["{}MA_vol".format(k)]
    return df_result.sort_index(ascending=False)


def k_line(k:int, df:pd.DataFrame):
    if df.shape[1] != 6:
        raise ValueError("df.shape[1] {}!=6".format(df.shape[1]))

    if not {"open", "high", "low", "close", "vol", "amt"}.issubset(set(
            df.columns)):
        raise ValueError("[\"open\", \"high\", \"low\", \"close\", \"vol\", "
                         "\"amt\"\] is not a subset of {}".format(set(
            df.columns)))

    df_result = pd.DataFrame(index=df.index)
    df = df.sort_index(ascending=True)

    df_result["{}k_open".format(k)] = pd.Series(df["open"].iloc[:-k + 1],
                                                index=df.index[k - 1:])
    df_result["{}k_high".format(k)] = df["high"].rolling(k).max()
    df_result["{}k_low".format(k)] = df["low"].rolling(k).min()
    df_result["{}k_close".format(k)]=df["close"]
    df_result["{}k_vol".format(k)] = df["vol"].rolling(k).mean()
    df_result["{}k_amt".format(k)] = df["amt"].rolling(k).mean()

    return df_result


def prepare_stock_d(df_stck_d):
    df_stck_d = df_stck_d.set_index(["date"]).sort_index(ascending=False)
    df_stck_d = df_stck_d[
        ["code", "open", "high", "low", "close", "vol", "amt", "adj_factor"]]
    return df_stck_d


def prepare_index_d(df_idx_d):
    df_idx_d = df_idx_d.set_index("date").sort_index(ascending=False)
    return df_idx_d


def prepare_each_stock(df_stock_d, qfq_type="hfq"):
    if qfq_type and qfq_type not in ["hfq","qfq"]:
        raise ValueError("qfq_type {} is not supported".format(qfq_type))

    df_stock_d = df_stock_d.copy()
    fq_cols = ["open", "high", "low", "close","vol"]

    # 原始数据
    for col in fq_cols:
        df_stock_d[col + "0"] = df_stock_d[col]

    # 前复权
    if qfq_type=="qfq":
        fq_factor = np.array(df_stock_d["adj_factor"]
                             / df_stock_d["adj_factor"].iloc[0])
    else:
        fq_factor = df_stock_d["adj_factor"]

    # print(fq_factor.shape)
    fq_factor = np.array(fq_factor).reshape(-1, 1) * np.ones(
        (1, len(fq_cols)))

    # Deal with open,high,low,close.
    df_stock_d.loc[:, fq_cols[:4]] = df_stock_d[fq_cols[:4]] * fq_factor[:, :4]
    # Deal with vol.
    df_stock_d.loc[:, fq_cols[4]] = df_stock_d[fq_cols[4]] / fq_factor[:, 0]
    # Calculate stocks' avg day prices.
    # vol's unit is "手", while amt's unit is "1000yuan", so 10 is multiplied.
    df_stock_d["avg"] = df_stock_d["amt"]/df_stock_d["vol"]*10

    return df_stock_d


def proc_stck_d(df_stck_d, stock_pool=None,targets=None,start=None):
    df_stck_d = prepare_stock_d(df_stck_d)

    df_stck_list = []
    cols_move = ["open", "high", "low", "close", "vol","amt"]
    cols_roll = ["open", "high", "low", "close", "vol","amt"]
    cols_k_line = ["open", "high", "low", "close","vol", "amt"]
    cols_fq = ["open", "high", "low", "close","avg"]

    move_upper_bound = 6
    move_mv_list = np.arange(1, move_upper_bound)

    candle_stick_mv_list = np.arange(0,move_upper_bound)

    kma_k_list = [3,5, 10, 20, 60, 120, 250]
    kma_mv_list = np.arange(0, move_upper_bound)

    k_line_k_list = kma_k_list
    k_line_mv_list = np.arange(0, move_upper_bound)

    rolling_k_list = np.array(kma_k_list,dtype=int)*-1

    cols_not_in_X = None

    for i,(code, df) in enumerate(df_stck_d.groupby("code")):
        if stock_pool and code not in stock_pool:
            continue

        # Initialize df.
        df = df.sort_index(ascending=False)
        df = prepare_each_stock(df)

        df_tomorrow = move(-1, df, ["open", "high", "low", "close"])

        df_qfq = df[cols_fq] / df["adj_factor"].iloc[0]
        df_qfq.columns = ["qfq_" + col for col in cols_fq]
        df_tomorrow_qfq = move(-1, df_qfq)

        df_targets_list = []
        for t in targets:
            pred_period = t["period"]
            if t["fun"]=="min":
                df_target = rolling(t["fun"], pred_period, move(-1, df, cols=t["col"]))
            elif t["fun"]=="max":
                # df_target = rolling(t["fun"],pred_period - 1, move(-2, df, cols=t["col"]))
                df_target = rolling(t["fun"], pred_period, move(-1, df, cols=t["col"]))
            elif t["fun"]=="mean":
                # df_target = rolling(t["fun"], pred_period - 1, move(-2, df, cols=t["col"]))
                df_target = rolling(t["fun"], pred_period, move(-1, df, cols=t["col"]))

                # p1 = (pred_period - 1) // 3
                # p2 = p1
                # p3 = pred_period - 1 - p1 - p2
                # df_period_mean1 = rolling(t["fun"], p1, move(-2, df, t["col"]))
                # df_period_mean2 = rolling(t["fun"], p2, move(-2 - p1, df, t["col"]))
                # df_period_mean3 = rolling(t["fun"], p3, move(-2 - p1 - p2, df, t["col"]))
                # df_targets_list.extend([df_period_mean1,df_period_mean2,df_period_mean3])
            else:
                raise ValueError("Fun type {} is not supported!".format(t["fun"]))
            df_targets_list.append(df_target)

        df_move_list = [move(i, df, cols_move) for i in move_mv_list]
        df_move_change_list = [change_rate(df_move,df[cols_move])
                               for df_move in df_move_list]

        df_candle_stick = candle_stick(df[cols_fq])
        df_move_candle_list = [move(i,df_candle_stick) for i in candle_stick_mv_list]

        df_1ma = k_MA(1,df[["vol","amt"]])
        df_kma_change_list = [change_rate(k_MA(k,df[["vol","amt"]]),df_1ma)
                       for k in kma_k_list]
        df_move_kma_change_list = [move(mv,df_kma_change)
            for df_kma_change in df_kma_change_list
            for mv in kma_mv_list
        ]

        df_k_line_list = [(k,k_line(k,df[cols_k_line])) for k in k_line_k_list]
        df_change_move_k_line_list = [change_rate(move(k*mv,df_k_line),
                                                  df[cols_k_line])
                                      for k,df_k_line in df_k_line_list
                                      for mv in k_line_mv_list]


        df_rolling_change_list = [
            change_rate(rolling(rolling_type, days=days, df=df, cols=cols_roll),
                        df[cols_roll],
                        )
            for days in rolling_k_list
            for rolling_type in ["max","min","mean"]]

        df_not_in_X = pd.concat(
            [df_qfq,df_tomorrow,df_tomorrow_qfq]+df_targets_list, axis=1, sort=False)

        # df_stck = pd.concat(
        #     [df] + df_move_change_list
        #     # + df_move_candle_list
        #     # + df_move_kma_change_list
        #     + df_rolling_change_list
        #     # + df_change_move_k_line_list
        #     + [df_not_in_X],
        #     axis=1,
        #     sort=False)

        df_stck = pd.concat([df] + df_move_change_list
                            + df_move_candle_list
                            + df_move_kma_change_list
                            + df_rolling_change_list
                            + df_change_move_k_line_list
                            + [df_not_in_X], axis=1, sort=False)

        df_stck_list.append(df_stck[df_stck.index>=start])

        if not cols_not_in_X:
            cols_not_in_X = list(df_not_in_X.columns)

        if i%10==0:
            print("Finish preparing {} stocks.".format(i))

    df_stck_d_all = pd.concat(df_stck_list, sort=False)

    print("count stck", len(
        df_stck_d_all["code"][df_stck_d_all.index >= "2018-01-01"].unique()))
    print(df_stck_d_all.shape)

    df_stck_d_all.index.name = "date"

    return df_stck_d_all, cols_not_in_X


def proc_idx_d(df_idx_d: pd.DataFrame, start=None):
    df_idx_d = prepare_index_d(df_idx_d)
    cols_move = ["open", "high", "low", "close", "vol"]
    cols_roll = cols_move

    rolling_k_list = [-3, -5, -10, -20, -60, -120, -250, -500]

    df_idx_list = []
    for name, group in df_idx_d.groupby("code"):
        group = group.sort_index(ascending=False)
        del group["code"]

        # print("group",group.index.name)
        df_move_list = [
            change_rate(move(i, group, cols_move),group[cols_move]) for i in
            range(1, 6)]
        df_rolling_list = [
            (change_rate(rolling("max",days, group,["high", "vol"]),
                         group[["high", "vol"]],),
             change_rate(rolling("min",days, group,["low", "vol"]),
                         group[["low", "vol"]],),
             change_rate(rolling("mean",days, group,["open", "close", "vol"]),
                         group[["open", "close", "vol"]],))
            for days in rolling_k_list
        ]

        df_rolling_flat_list = []
        for df_rolling_group in df_rolling_list:
            df_rolling_flat_list.extend(df_rolling_group)

        tmp_list = [group] + df_move_list + df_rolling_flat_list
        tmp = pd.concat(tmp_list, axis=1, sort=False)
        # print("tmp_list",[t.index.name for t in tmp_list],pd.concat(
        #     tmp_list,axis=1,sort=False).index.name)
        # print("tmp",tmp.index.name)
        df_idx_list.append(_prefix(name, tmp[tmp.index>=start]))

    # print("df_idx_list:",df_idx_list[0].index.name)

    df_idx_d = pd.concat(df_idx_list, axis=1, sort=False)
    # print("df_idx_d:",df_idx_d.index.name)
    df_idx_d.index.name="date"
    return df_idx_d


def proc_stock_basic(df_stock_basic:pd.DataFrame):
    cols_category = ["area", "industry", "market", "exchange", "is_hs"]
    df_stock_basic = df_stock_basic[["code"] + cols_category].copy()
    print(df_stock_basic[df_stock_basic[cols_category].isna().any(axis=1)][
              cols_category])
    df_stock_basic.loc[:,cols_category] = df_stock_basic[cols_category].fillna("")

    enc = sk.preprocessing.OrdinalEncoder()
    df_stock_basic.loc[:,cols_category] = enc.fit_transform(df_stock_basic[cols_category])
    return df_stock_basic,cols_category,enc


def prepare_data(cursor, targets=None, start=None, lowerbound=None, stock_pool=None):
    print("start:",start,"\tlowerbound:", lowerbound)

    # Prepare df_stock_basic
    df_stock_basic = dbop.create_df(cursor, const.STOCK_BASIC[const.TABLE])
    df_stock_basic, cols_category, enc = proc_stock_basic(df_stock_basic)
    print(df_stock_basic.iloc[:10])

    # Prepare df_stock_d_FE
    df_stock_d = dbop.create_df(cursor, const.STOCK_DAY[const.TABLE], lowerbound)
    print("min_date:", min(df_stock_d.date))
    df_stock_d_FE, cols_future = proc_stck_d(df_stock_d,
                                             stock_pool=stock_pool,
                                             targets=targets,
                                             start=start)
    df_stock_d_FE = df_stock_d_FE[df_stock_d_FE.index>=start]
    print(df_stock_d_FE.shape)
    print(df_stock_d_FE.index.name)

    # Prepare df_index_d_FE
    df_index_d = dbop.create_df(cursor, const.INDEX_DAY[const.TABLE], lowerbound)
    df_index_d_FE = proc_idx_d(df_index_d,start=start)
    df_index_d_FE = df_index_d_FE[df_index_d_FE>=start]
    print(df_index_d_FE.shape, len(df_index_d_FE.index.unique()))
    print(df_index_d_FE.index.name)

    # Merge three df.
    df_all = df_stock_d_FE.join(df_index_d_FE, how="left")
    df_all.index.name = "date"
    df_all = df_all.reset_index()\
        .merge(df_stock_basic, on="code",how="left")\
        .set_index(["date"])

    cols_not_for_model = list(df_stock_basic.columns.difference(
        cols_category+["code"]))
    print(df_all.shape)

    return df_all, cols_future, cols_category, cols_not_for_model, enc


def feature_select(X, y):
    import sklearn.ensemble as ensemble
    clf = ensemble.ExtraTreesClassifier(random_state=0)
    clf.fit(X, y)
    import sklearn.feature_selection as fselect
    model = fselect.SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print("selected feature number:", X_new.shape)

    return X_new, model