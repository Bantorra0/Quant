import matplotlib.pyplot as plt
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


def _move(days, df: pd.DataFrame, cols=None, prefix=True):
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


def _rolling(rolling_type, days, df: pd.DataFrame, cols, move=0,
             has_prefix=True):
    _check_int(days)
    cols = _make_iterable(cols)

    period = abs(days)
    if rolling_type == "max":
        df_rolling = df[cols].rolling(window=abs(days)).max()
    elif rolling_type == "min":
        df_rolling = df[cols].rolling(window=abs(days)).min()
    elif rolling_type == "mean":
        df_rolling = df[cols].rolling(window=abs(days)).max()
    else:
        raise ValueError(
            "rolling_type='{}' is not supported.".format(rolling_type))

    if move != 0:
        df_rolling = _move(move, df_rolling)
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

    if has_prefix:
        return _prefix(pre, df_rolling)
    else:
        return df_rolling


def _rolling_max(days, df: pd.DataFrame, cols, move=0, has_prefix=True):
    _check_int(days)
    cols = _make_iterable(cols)

    period = abs(days)
    df_rolling = df[cols].rolling(window=abs(days)).max()
    if move != 0:
        # print("--------",move)
        # print(df_rolling[df["code"] == "600887.SH"]["high"].iloc[:30])
        df_rolling = _move(move,
                           df_rolling)  # print(df_rolling[df["code"] == "600887.SH"]["f1mv_high"].iloc[:30])
    n = len(df_rolling)
    idxes = df_rolling.index
    if days > 0:
        pre = "f" + str(abs(days)) + "max"
        df_rolling = df_rolling.iloc[period - 1:n]
        df_rolling.index = idxes[
                           period - 1:n]  # df_rolling = df_rolling.iloc[period-1:n+move]  # df_rolling.index = df.index[period-1-move:n]
    else:
        pre = "p" + str(abs(days)) + "max"
        df_rolling = df_rolling.iloc[period - 1:n]
        if n - period + 1 >= 0:
            df_rolling.index = idxes[:n - period + 1]

        # df_rolling = df_rolling.iloc[period-1+move:n]  # df_rolling.index = df.index[:n-period+1-move]

    if has_prefix:
        return _prefix(pre, df_rolling)
    else:
        return df_rolling


def _rolling_min(days, df: pd.DataFrame, cols, move=0, has_prefix=True):
    _check_int(days)
    cols = _make_iterable(cols)

    period = abs(days)
    df_rolling = df[cols].rolling(window=abs(days)).min()
    if move != 0:
        # print("--------",move)
        # print(df_rolling[df["code"] == "600887.SH"]["high"].iloc[:30])
        df_rolling = _move(move,
                           df_rolling)  # print(df_rolling[df["code"] == "600887.SH"]["f1mv_high"].iloc[:30])
    n = len(df_rolling)
    idxes = df_rolling.index
    if days > 0:
        pre = "f" + str(abs(days)) + "min"
        df_rolling = df_rolling.iloc[period - 1:n]
        df_rolling.index = idxes[period - 1:n]
    else:
        pre = "p" + str(abs(days)) + "min"
        df_rolling = df_rolling.iloc[period - 1:n]
        if n - period + 1 >= 0:
            df_rolling.index = idxes[:n - period + 1]

    if has_prefix:
        return _prefix(pre, df_rolling)
    else:
        return df_rolling


def _rolling_mean(days, df: pd.DataFrame, cols, move=0, has_prefix=True):
    _check_int(days)
    cols = _make_iterable(cols)

    period = abs(days)
    df_rolling = df[cols].rolling(window=abs(days)).mean()
    if move != 0:
        df_rolling = _move(move, df_rolling)
    n = len(df_rolling)
    idxes = df_rolling.index
    if days > 0:
        pre = "f" + str(abs(days)) + "mean"
        df_rolling = df_rolling.iloc[period - 1:n]
        df_rolling.index = idxes[period - 1:n]
    else:
        pre = "p" + str(abs(days)) + "mean"
        df_rolling = df_rolling.iloc[period - 1:n]
        if n - period + 1 >= 0:
            df_rolling.index = idxes[:n - period + 1]

    if has_prefix:
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
    df1.columns = df2.columns
    df3 = (df2 - df1) / df1
    df3 = _prefix("change_rate", df3)
    return df3


def create_df(cursor, table_name, start=None):
    if start:
        sql_select = "select * from {0} where date>='{1}'".format(table_name,
                                                                start)
    else:
        sql_select = "select * from {0}".format(table_name)
    cursor.execute(sql_select)
    df = pd.DataFrame(cursor.fetchall())
    df.columns = dbop.cols_from_cur(cursor)
    return df


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
    qfq_cols = ["open", "high", "low", "close"]

    for col in qfq_cols:
        df_stck[col+"0"] = df_stck[col]

    # 后复权
    if qfq_type=="qfq":
        qfq_factor = np.array(df_stck["adj_factor"]
                          / df_stck["adj_factor"].iloc[0])

    # print(qfq_factor.shape)
    qfq_factor = np.array(df_stck["adj_factor"]).reshape(-1, 1) * np.ones(
        (1, len(qfq_cols)))
    # print(df_stck[qfq_cols].dtypes)
    # print(qfq_factor.shape, qfq_factor.dtype)
    # print(df_stck[qfq_cols]/qfq_factor)
    df_stck.loc[:, qfq_cols] = df_stck[qfq_cols] * qfq_factor

    return df_stck


def proc_stck_d(df_stck_d, pred_period=10):
    df_stck_d = prepare_stck_d(df_stck_d)

    df_stck_list = []
    cols_move = ["open", "high", "low", "close", "amt"]
    cols_roll = ["open", "high", "low", "close", "amt"]
    cols_future = None
    for code, df in df_stck_d.groupby("code"):
        df = df.sort_index(ascending=False)
        df = prepare_each_stck(df)
        df_label_min = _rolling_min(pred_period, df, "low", move=-1)
        df_label_max = _rolling_max(pred_period - 1, df, "high", move=-2)
        p1 = (pred_period - 1) // 3
        p2 = p1
        p3 = pred_period - 1 - p1 - p2
        df_label_mean1 = _rolling_mean(p1, df, "open", move=-2)
        df_label_mean2 = _rolling_mean(p2, df, "open", move=-2 - p1)
        df_label_mean3 = _rolling_mean(p3, df, "open", move=-2 - p1 - p2)

        # print(df_label_min.columns)
        df_tomorrow = _move(-1, df, ["open", "high", "low", "close"])

        # df_label_min = _rolling_min(pred_period,df,"low")

        # if code == "000002.SZ":
        #     tmp = _rolling_min(-5,df,cols_roll).loc["2018-08-07"]
        #     print(tmp)
        df_move_list = [change_rate(df[cols_move], _move(i, df, cols_move)) for
                        i in range(1, 6)]
        df_rolling_list = [(change_rate(df[cols_roll],
                                        _rolling_max(i, df, cols_roll)),
                            change_rate(df[cols_roll],
                                        _rolling_min(i, df, cols_roll)),
                            change_rate(df[cols_roll],
                                        _rolling_mean(i, df, cols_roll))) for i
                           in [-5, -10, -20, -60, -120, -250]]

        df_roll_flat_list = []
        for df_rolling_group in df_rolling_list:
            df_roll_flat_list.extend(df_rolling_group)

        df_labels = pd.concat(
            [df_tomorrow, df_label_max, df_label_min, df_label_mean1,
             df_label_mean2, df_label_mean3], axis=1, sort=False)
        df_stck = pd.concat(
            [df] + df_move_list + df_roll_flat_list + [df_labels], axis=1,
            sort=False)
        df_stck_list.append(df_stck)

        if not cols_future:
            cols_future = list(df_labels)
        # print(tmp.shape)
        # print(tmp[tmp[col_label].isnull()])
        # if code == "002217.SZ":
        #     print(df[df.index == "2018-01-02"])
        #     print(df_stck[df_stck.index == "2018-01-02"])

    df_stck_d_all = pd.concat(df_stck_list, sort=False)
    # for df in df_stck_list:
    #     print(df["code"].unique(), df.shape)
    #     print(df["code"].unique(), df[df.index >= "2018-01-01"].shape)
    print("count stck", len(
        df_stck_d_all["code"][df_stck_d_all.index >= "2018-01-01"].unique()))
    print(df_stck_d_all.shape)

    return df_stck_d_all, cols_future


def proc_idx_d(df_idx_d: pd.DataFrame):
    df_idx_d = prepare_idx_d(df_idx_d)
    cols_move = ["open", "high", "low", "close", "vol"]
    cols_roll = cols_move

    df_idx_list = []
    for name, group in df_idx_d.groupby("code"):
        group = group.sort_index(ascending=False)
        del group["code"]
        df_move_list = [
            change_rate(group[cols_move], _move(i, group, cols_move)) for i in
            range(1, 6)]
        df_rolling_list = [(change_rate(group[["high", "vol"]],
                                        _rolling_max(i, group,
                                                     ["high", "vol"])),
                            change_rate(group[["low", "vol"]],
                                        _rolling_min(i, group,
                                                     ["low", "vol"])),
                            change_rate(group[["open", "close", "vol"]],
                                        _rolling_mean(i, group,
                                                      ["open", "close",
                                                       "vol"]))) for i in
                           [-5, -10, -20, -60, -120, -250, -500]]

        df_roll_flat_list = []
        for df_rolling_group in df_rolling_list:
            df_roll_flat_list.extend(df_rolling_group)

        tmp_list = [group] + df_move_list + df_roll_flat_list
        tmp = pd.concat(tmp_list, axis=1, sort=False)
        df_idx_list.append(_prefix(name, tmp))

    df_idx_d = pd.concat(df_idx_list, axis=1, sort=False)
    return df_idx_d


def prepare_data(cursor, pred_period=10, start=None):
    stock_day, index_day = constants.STOCK_DAY[clct.TABLE], constants.INDEX_DAY[
        clct.TABLE]
    print("start:",start)
    df_stck_d = create_df(cursor, stock_day, start)
    print("min_date",min(df_stck_d.date))
    df_idx_d = create_df(cursor, index_day, start)

    df_stck_d_all, cols_future = proc_stck_d(df_stck_d,
                                             pred_period=pred_period)
    print(df_stck_d_all.shape)

    df_idx_d = proc_idx_d(df_idx_d)
    print(df_idx_d.shape, len(df_idx_d.index.unique()))
    df_all = df_stck_d_all.join(df_idx_d)
    print(df_all.shape)

    # print(df_all[(df_all.index == "2018-01-02") & (
    #             df_all["code"] == "002217.SZ")])

    return df_all, cols_future


def y_distribution(y):
    y = y.copy().dropna()
    # print distribution of y
    print("before", sum(y < 0))
    print("y<-0.5:", sum(y < -0.5))
    for i in range(-5, 5):
        tmp1 = ((i * 0.1) <= y)
        tmp2 = (y < ((i + 1) * 0.1))
        if len(tmp1) == 0 or len(tmp2) == 0:
            tmp = [False]
        else:
            tmp = tmp1 & tmp2
        print("{0:.2f}<=y<{1:.2f}:".format(i * 0.1, (i + 1) * 0.1), sum(tmp))
    print("y>0.5", sum(y > 0.5))
    print("after", sum(y < 0))
    plt.figure()
    plt.hist(y, bins=np.arange(-10, 11) * 0.1)


def gen_y(df_all: pd.DataFrame, pred_period=10,threshold=0.1,
          label_type=None):
    if label_type and (label_type not in ["inc", "dec"]):
        raise ValueError("label_type {} is not supported!".format(label_type))

    y = df_all["f{}max_f2mv_high".format(pred_period - 1)] / df_all["f1mv_open"] - 1
    y_distribution(y)

    # print("--------------")
    # print(y[y.isna() & (df_all["f1mv_high"] == df_all["f1mv_low"])])
    y[y.notnull() & (df_all["f1mv_high"] == df_all["f1mv_low"])] = 0
    print("过滤涨停项：", sum(df_all["f1mv_high"] == df_all["f1mv_low"]))

    return label(y,threshold,label_type)


def label(y, threshold=0.1, label_type=None):
    if label_type == "dec":
        y = -y

    if label_type:
        y[y > threshold] = 1
        y[y <= threshold] = 0

    return y


def feature_select(X, y):
    import sklearn.ensemble as ensemble
    clf = ensemble.ExtraTreesClassifier(random_state=0)
    clf.fit(X, y)
    import sklearn.feature_selection as fselect
    model = fselect.SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print("selected feature number:", X_new.shape)

    return X_new, model


def drop_null(X, y):
    Xy = np.concatenate((np.array(X), np.array(y).reshape(-1, 1)), axis=1)
    Xy = pd.DataFrame(Xy, index=X.index).dropna()
    X = Xy.iloc[:, :-1].copy()
    y = Xy.iloc[:, -1].copy()
    return X, y


def main():
    db_type = "sqlite3"

    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    pred_period=20
    df_all,cols_future = prepare_data(cursor,pred_period=pred_period,start="2011-01-01")

    # test
    # df_test = df_all[df_all["code"]=="600887.SH"]
    # basic_cols = ["open", "high", "low", "close", "amt", "adj_factor"]
    # derived_cols = ['change_rate_p1mv_open', 'change_rate_p1mv_high',
    #                 'change_rate_p1mv_low', 'change_rate_p1mv_close',
    #                 'change_rate_p1mv_amt', 'change_rate_p3mv_open',
    #                 'change_rate_p3mv_high', 'change_rate_p3mv_low',
    #                 'change_rate_p3mv_close', 'change_rate_p3mv_amt',
    #                 'change_rate_p5mv_open', 'change_rate_p5mv_high',
    #                 'change_rate_p5mv_low', 'change_rate_p5mv_close',
    #                 'change_rate_p5mv_amt', 'change_rate_p5max_open',
    #                 'change_rate_p5max_high', 'change_rate_p5max_low',
    #                 'change_rate_p5max_close', 'change_rate_p5max_amt',
    #                 'change_rate_p5min_open', 'change_rate_p5min_high',
    #                 'change_rate_p5min_low', 'change_rate_p5min_close',
    #                 'change_rate_p5min_amt', 'change_rate_p5mean_open',
    #                 'change_rate_p5mean_high', 'change_rate_p5mean_low',
    #                 'change_rate_p5mean_close', 'change_rate_p5mean_amt',
    #                 'change_rate_p20max_open', 'change_rate_p20max_high',
    #                 'change_rate_p20max_low', 'change_rate_p20max_close',
    #                 'change_rate_p20max_amt', 'change_rate_p20min_open',
    #                 'change_rate_p20min_high', 'change_rate_p20min_low',
    #                 'change_rate_p20min_close', 'change_rate_p20min_amt',
    #                 'change_rate_p20mean_open', 'change_rate_p20mean_high',
    #                 'change_rate_p20mean_low', 'change_rate_p20mean_close',
    #                 'change_rate_p20mean_amt', 'f1mv_open', 'f1mv_high',
    #                 'f1mv_low', 'f1mv_close', 'f20max_f1mv_high',
    #                 'sz50_open', 'sz50_high', 'sz50_low', 'sz50_close',
    #                 'sz50_vol', 'sz50_change_rate_p1mv_open',
    #                 'sz50_change_rate_p1mv_high',
    #                 'sz50_change_rate_p1mv_low',
    #                 'sz50_change_rate_p1mv_close',
    #                 'sz50_change_rate_p1mv_vol']
    #
    # test_cols = basic_cols + derived_cols
    # print(test_cols)
    # df_test[test_cols].sort_index(ascending=False).iloc[:100].to_excel(
    #     "test_data.xlsx",header=True,index=True)




    # # test
    # df_test_list = []
    # for code in df_all["code"].unique()[:3]:
    #     df = df_all[df_all["code"]==code].sort_index(
    #         ascending=False).iloc[:50]
    #     print(df)
    #     df_test_list.append(df)
    # pd.concat(df_test_list).to_excel("test_data.xlsx",header=True,index=True)
    #
    #
    import xgboost.sklearn as xgb
    import lightgbm.sklearn as lgbm
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    import time
    import sklearn.preprocessing as preproc

    period = (df_all.index >= "2014-01-01")
    df_all = df_all[period]

    df_all = df_all[df_all["amt"]!=0]

    y = gen_y(df_all, threshold=0.15, pred_period=pred_period)
    print("null:",sum(y.isnull()))

    features = df_all.columns.difference(cols_future+["code"])


    X = df_all[features]


    # X,y = drop_null(X,y)
    X = X[y.notnull()]
    X_full = df_all[y.notnull()]
    print("full and X",X.shape,X_full.shape)
    y = y.dropna()
    print(X.shape,y.shape)
    print("total positive", sum(y))

    condition = (X.index >= "2018-01-01")
    X_train, y_train = X[~condition], y[~condition]
    X_test, y_test = X[condition], y[condition]

    print(X_test.shape,y_test.shape)
    print("test positive:", sum(y_test))

    X_train_full = X_full.loc[condition]
    X_test_full = X_full.loc[condition]

    print(X_test_full.shape,X_test.shape)
    print(X_test_full[(X_test_full.index == "2018-01-02") & (X_test_full["code"]=="002217.SZ")].shape)


    # print(X_test_full["code"].iloc[np.array(y_test == 1)])
    # print(X_test_full[X_test_full["code"]=="002217.SZ"])

    # # scaler = preproc.StandardScaler()
    # # X_train = scaler.fit_transform(X_train)
    # # X_test = scaler.transform(X_test)
    #
    # # X_train,selector = feature_select(X_train,y_train)
    # # X_test = selector.transform(X_test)
    #
    #
    scale_pos_weight = sum(y==0)/sum(y==1)

    clfs = [
        lgbm.LGBMClassifier(n_estimators=300, scale_pos_weight=0.1,
                            num_leaves=100, max_depth=8, random_state=0),
        xgb.XGBClassifier(n_estimators=300, scale_pos_weight=0.1,
                          max_depth=5,
                          random_state=0,),
    ]

    y_prd_list = []
    colors = ["r", "b"]
    for clf, c in zip(clfs, colors):
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        y_prd_list.append([clf, t2 - t1, clf.predict_proba(X_test), c])

    for clf, t, y_prd_prob, c in y_prd_list:
        y_prd = np.where(y_prd_prob[:, 0] < 0.25, 1, 0)
        print(clf.classes_)
        print(y_prd.shape, sum(y_prd))

        print(X_test_full["code"].iloc[y_prd==1])
        # print(X_test_full["code"])

        print("accuracy", metrics.accuracy_score(y_test, y_prd))
        print("precison", metrics.precision_score(y_test, y_prd))
        print("recall", metrics.recall_score(y_test, y_prd))
        precision, recall, _ = metrics.precision_recall_curve(y_test, y_prd_prob[:, 1])

        plt.figure()
        plt.title(clf.__class__)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.plot(recall, precision, color=c)
        print(clf, t)

    plt.show()


if __name__ == '__main__':
    main()