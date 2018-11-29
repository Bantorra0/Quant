import numpy as np
import pandas as pd

import collect as clct
import constants
import db_operations as dbop


def _check_int(arg):
    if type(arg) != int:
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


def change_rate(df1: pd.DataFrame, df2: pd.DataFrame, cols1=None, cols2=None):
    if cols1:
        df1 = df1[cols1].copy()
    if cols2:
        df2 = df2[cols2].copy()

    if df1.shape[1] != df2.shape[1]:
        raise ValueError(
            "Column length not the same:{0}!={1}".format(df1.shape[1],
                                                         df2.shape[1]))

    df1 = df1.copy()
    # Make sure columns of df1 and df2 are the same, because operations are
    # based on index and columns.
    df1.columns = df2.columns
    df3 = (df2 - df1) / df1
    df3 = _prefix("change_rate", df3)
    return df3


def prepare_stck_d(df_stck_d):
    df_stck_d = df_stck_d.set_index(["date"]).sort_index(ascending=False)
    df_stck_d = df_stck_d[
        ["code", "open", "high", "low", "close", "vol", "amt", "adj_factor"]]
    return df_stck_d


def prepare_idx_d(df_idx_d):
    df_idx_d = df_idx_d.set_index("date").sort_index(ascending=False)
    return df_idx_d


def prepare_each_stck(df_stck, qfq_type="hfq"):
    if qfq_type and qfq_type not in ["hfq","qfq"]:
        raise ValueError("qfq_type {} is not supported".format(qfq_type))

    df_stck = df_stck.copy()
    fq_cols = ["open", "high", "low", "close"]

    # 原始数据
    for col in fq_cols:
        df_stck[col+"0"] = df_stck[col]

    # 后复权
    if qfq_type=="qfq":
        qfq_factor = np.array(df_stck["adj_factor"]
                          / df_stck["adj_factor"].iloc[0])

    # print(qfq_factor.shape)
    qfq_factor = np.array(df_stck["adj_factor"]).reshape(-1, 1) * np.ones(
        (1, len(fq_cols)))

    df_stck.loc[:, fq_cols] = df_stck[fq_cols] * qfq_factor

    return df_stck


def proc_stck_d(df_stck_d, stock_pool=None,targets=None):
    df_stck_d = prepare_stck_d(df_stck_d)

    df_stck_list = []
    cols_move = ["open", "high", "low", "close", "amt"]
    cols_roll = ["open", "high", "low", "close", "amt"]
    fq_cols = ["open", "high", "low", "close"]
    cols_not_in_X = None
    for code, df in df_stck_d.groupby("code"):
        if stock_pool and code not in stock_pool:
            continue

        df = df.sort_index(ascending=False)
        df = prepare_each_stck(df)
        df_tomorrow = move(-1, df, ["open", "high", "low", "close"])

        df_targets_list = []
        for t in targets:
            pred_period = t["period"]
            if t["fun"]=="min":
                df_target = rolling(t["fun"], pred_period, move(-1, df, cols=t["col"]))
                df_targets_list.append(df_target)
            elif t["fun"]=="max":
                df_target = rolling(t["fun"],pred_period - 1, move(-2, df, cols=t["col"]))
                df_targets_list.append(df_target)
            elif t["fun"]=="mean":
                p1 = (pred_period - 1) // 3
                p2 = p1
                p3 = pred_period - 1 - p1 - p2
                df_period_mean1 = rolling(t["fun"], p1, move(-2, df, t["col"]))
                df_period_mean2 = rolling(t["fun"], p2, move(-2 - p1, df, t["col"]))
                df_period_mean3 = rolling(t["fun"], p3, move(-2 - p1 - p2, df, t["col"]))
                df_targets_list.extend([df_period_mean1,df_period_mean2,df_period_mean3])

        df_move_list = [change_rate(df[cols_move], move(i, df, cols_move)) for
                        i in range(1, 6)]

        df_qfq = df[fq_cols] / df["adj_factor"].iloc[0]
        df_qfq.columns = ["qfq_"+col for col in fq_cols]
        df_tomorrow_qfq = move(-1, df_qfq)

        df_rolling_list = [
            change_rate(df[cols_roll],
                        rolling(rolling_type, days=days, df=df, cols=cols_roll))
            for days in [-5, -10, -20, -60, -120, -250]
            for rolling_type in ["max","min","mean"]]

        df_not_in_X = pd.concat(
            [df_qfq,df_tomorrow,df_tomorrow_qfq]+df_targets_list, axis=1, sort=False)
        df_stck = pd.concat(
            [df] + df_move_list + df_rolling_list + [df_not_in_X], axis=1,
            sort=False)
        df_stck_list.append(df_stck)

        if not cols_not_in_X:
            cols_not_in_X = list(df_not_in_X.columns)

    df_stck_d_all = pd.concat(df_stck_list, sort=False)

    print("count stck", len(
        df_stck_d_all["code"][df_stck_d_all.index >= "2018-01-01"].unique()))
    print(df_stck_d_all.shape)

    return df_stck_d_all, cols_not_in_X


def proc_idx_d(df_idx_d: pd.DataFrame):
    df_idx_d = prepare_idx_d(df_idx_d)
    cols_move = ["open", "high", "low", "close", "vol"]
    cols_roll = cols_move

    df_idx_list = []
    for name, group in df_idx_d.groupby("code"):
        group = group.sort_index(ascending=False)
        del group["code"]
        df_move_list = [
            change_rate(group[cols_move], move(i, group, cols_move)) for i in
            range(1, 6)]
        df_rolling_list = [
            (change_rate(group[["high", "vol"]],
                         rolling("max",days, group,["high", "vol"])),
             change_rate(group[["low", "vol"]],
                         rolling("min",days, group,["low", "vol"])),
             change_rate(group[["open", "close", "vol"]],
                         rolling("mean",days, group,["open", "close", "vol"])))
            for days in [-5, -10, -20, -60, -120, -250, -500]
        ]

        df_rolling_flat_list = []
        for df_rolling_group in df_rolling_list:
            df_rolling_flat_list.extend(df_rolling_group)

        tmp_list = [group] + df_move_list + df_rolling_flat_list
        tmp = pd.concat(tmp_list, axis=1, sort=False)
        df_idx_list.append(_prefix(name, tmp))

    df_idx_d = pd.concat(df_idx_list, axis=1, sort=False)
    return df_idx_d


def prepare_data(cursor,targets=None, start=None, stock_pool=None):
    stock_day, index_day = constants.STOCK_DAY[clct.TABLE], constants.INDEX_DAY[
        clct.TABLE]
    print("start:",start)
    df_stck_d = dbop.create_df(cursor, stock_day, start)
    print("min_date",min(df_stck_d.date))
    df_idx_d = dbop.create_df(cursor, index_day, start)

    df_stck_d_all, cols_future = proc_stck_d(df_stck_d,
                                             stock_pool=stock_pool,
                                             targets=targets)
    print(df_stck_d_all.shape)

    df_idx_d = proc_idx_d(df_idx_d)
    print(df_idx_d.shape, len(df_idx_d.index.unique()))
    df_all = df_stck_d_all.join(df_idx_d)
    print(df_all.shape)

    return df_all, cols_future


def feature_select(X, y):
    import sklearn.ensemble as ensemble
    clf = ensemble.ExtraTreesClassifier(random_state=0)
    clf.fit(X, y)
    import sklearn.feature_selection as fselect
    model = fselect.SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print("selected feature number:", X_new.shape)

    return X_new, model