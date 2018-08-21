import db_operations as dbop
import df_operations as dfop
import collect as clct
import pandas as pd
import numpy as np


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


def _move(days, df: pd.DataFrame, cols):
    _check_int(days)
    cols = _make_iterable(cols)

    if days > 0:
        pre = "p{}move".format(abs(days))
        df_move = df[cols].iloc[days:].copy()
        df_move.index = df.index[:-days]
    else:
        pre = "f{}move".format(abs(days))
        df_move = df[cols].iloc[:days].copy()
        df_move.index = df.index[-days:]

    return _prefix(pre, df_move)


def _rolling_max(days, df: pd.DataFrame, cols):
    _check_int(days)
    cols = _make_iterable(cols)

    df_rolling = df[cols].rolling(window=abs(days)).max()
    if days > 0:
        cols_rolling = list(map(lambda s: "f" + str(abs(days)) + "max_" + str(s), cols))
    else:
        cols_rolling = list(map(lambda s: "p" + str(abs(days)) + "max_" + str(s), cols))
        df_rolling = df_rolling.dropna()
        df_rolling.index = df.index[:len(df_rolling)]

    df_rolling.columns = cols_rolling
    return df_rolling


def _rolling_min(days, df: pd.DataFrame, cols):
    _check_int(days)
    cols = _make_iterable(cols)

    df_rolling = df[cols].rolling(window=abs(days)).min()
    if days > 0:
        cols_rolling = list(map(lambda s: "f" + str(abs(days)) + "min_" + str(s),
                                cols))
    else:
        cols_rolling = list(map(lambda s: "p" + str(abs(days)) + "min_" + str(s),
                                cols))
        df_rolling = df_rolling.dropna()
        df_rolling.index = df.index[:len(df_rolling)]

    df_rolling.columns = cols_rolling
    return df_rolling


def _rolling_mean(days, df: pd.DataFrame, cols):
    _check_int(days)
    cols = _make_iterable(cols)

    df_rolling = df[cols].rolling(window=abs(days)).mean()
    if days > 0:
        cols_rolling = list(map(lambda s: "f" + str(abs(days)) + "mean_" + str(s), cols))
    else:
        cols_rolling = list(map(lambda s: "p" + str(abs(days)) + "mean_" + str(s), cols))
        df_rolling = df_rolling.dropna()
        df_rolling.index = df.index[:len(df_rolling)]

    df_rolling.columns = cols_rolling
    return df_rolling


def change_rate(df1: pd.DataFrame, df2: pd.DataFrame):
    if df1.shape[1] != df2.shape[1]:
        raise ValueError("Column length not the same:{0}!={1}".format(df1.shape[1], df2.shape[1]))

    df1 = df1.copy()
    df1.columns = df2.columns
    df3 = (df2 - df1) / df1
    df3 = _prefix("change_rate", df3)
    return df3


def create_df(cursor, table_name):
    sql_select_all = "select * from {}"
    cursor.execute(sql_select_all.format(table_name))
    df = pd.DataFrame(cursor.fetchall())
    df.columns = dbop.cols_from_cur(cursor)
    return df


def prepare_stck_d(df_stck_d):
    df_stck_d = df_stck_d.set_index(["date"]).sort_index(ascending=False)
    df_stck_d = df_stck_d[["code", "open", "high", "low", "close", "vol", "amt", "adj_factor"]]
    return df_stck_d


def prepare_idx_d(df_idx_d):
    df_idx_d = df_idx_d.set_index("date").sort_index(ascending=False)
    return df_idx_d


def prepare_each_stck(df_stck):
    df_stck = df_stck.copy()
    qfq_cols = ["open", "high", "low", "close"]
    qfq_factor = np.array(df_stck["adj_factor"] / df_stck["adj_factor"].iloc[0])
    print(qfq_factor.shape)
    qfq_factor = qfq_factor.reshape(-1, 1) * np.ones((1, len(qfq_cols)))
    print(df_stck[qfq_cols].dtypes)
    print(qfq_factor.shape, qfq_factor.dtype)
    # print(df_stck[qfq_cols]/qfq_factor)
    df_stck.loc[:, qfq_cols] = df_stck[qfq_cols] * qfq_factor
    print(qfq_factor[:30])
    return df_stck


def proc_stck_d(df_stck_d):
    df_stck_d = prepare_stck_d(df_stck_d)

    df_stck_list = []
    pred_period = 20
    cols_move = ["open", "high", "low", "close", "amt"]
    cols_roll = ["open", "high", "low", "close", "amt"]
    for _, df in df_stck_d.groupby("code"):
        df = prepare_each_stck(df)
        del df["code"]
        df_label_max = _rolling_max(pred_period, df, "high")
        # df_label_min = _rolling_min(pred_period,df,"low")

        df_move_list = [change_rate(df[cols_move], _move(i, df, cols_move)) for i in range(1, 6)]
        df_rolling_list = [(change_rate(df[cols_roll], _rolling_max(i, df, cols_roll)),
                            change_rate(df[cols_roll], _rolling_min(i, df, cols_roll)),
                            change_rate(df[cols_roll], _rolling_mean(i, df, cols_roll)))
                           for i in [-5, -10, -20]]

        df_roll_flat_list = []
        for df_rolling_group in df_rolling_list:
            df_roll_flat_list.extend(df_rolling_group)

        tmp = pd.concat([df] + df_move_list + df_roll_flat_list + [df_label_max],
                        axis=1, sort=False)
        df_stck_list.append(tmp)
        # print(tmp.shape)
        # print(tmp[tmp[col_label].isnull()])

    df_stck_d_all = pd.concat(df_stck_list, sort=False)

    return df_stck_d_all


def proc_idx_d(df_idx_d: pd.DataFrame):
    df_idx_d = prepare_idx_d(df_idx_d)
    cols_move = ["open", "high", "low", "close", "vol"]
    cols_roll = cols_move

    df_idx_list = []
    for name, group in df_idx_d.groupby("code"):
        del group["code"]
        df_move_list = [change_rate(group[cols_move], _move(i, group, cols_move)) for i in range(1, 6)]
        df_rolling_list = [(change_rate(group[cols_roll], _rolling_max(i, group, cols_roll)),
                            change_rate(group[cols_roll], _rolling_min(i, group, cols_roll)),
                            change_rate(group[cols_roll], _rolling_mean(i, group, cols_roll)))
                           for i in [-5, -10, -20]]

        df_roll_flat_list = []
        for df_rolling_group in df_rolling_list:
            df_roll_flat_list.extend(df_rolling_group)

        tmp_list = [group] + df_move_list + df_roll_flat_list
        tmp = pd.concat(tmp_list, axis=1, sort=False)
        df_idx_list.append(_prefix(name, tmp))

    df_idx_d = pd.concat(df_idx_list, axis=1, sort=False)
    return df_idx_d


def main():
    db_type = "sqlite3"

    # init_table(STOCK_DAY.TABLE_NAME, db_type)
    # collect_stock_day(STOCK_DAY.pools(),db_type)
    #
    # init_table(INDEX_DAY.TABLE_NAME, db_type)
    # collect_index_day(INDEX_DAY.pools(), db_type)

    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    stock_day, index_day = clct.STOCK_DAY[clct.TABLE], clct.INDEX_DAY[clct.TABLE]

    df_stck_d = create_df(cursor, stock_day)
    df_idx_d = create_df(cursor, index_day)

    df_stck_d_all = proc_stck_d(df_stck_d)
    print(df_stck_d_all.shape)
    col_label = df_stck_d_all.columns[-1]

    df_idx_d = proc_idx_d(df_idx_d)
    print(df_idx_d.shape)
    df_all = df_stck_d_all.join(df_idx_d)

    import xgboost.sklearn as xgb
    import lightgbm.sklearn as lgbm
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    import time

    clfs = [
        xgb.XGBClassifier(n_estimators=200, scale_pos_weight=0.1, max_depth=5, random_state=0),
        lgbm.LGBMClassifier(n_estimators=400, scale_pos_weight=0.166,
                            num_leaves=31,
                            max_depth=5,random_state=0)
    ]

    df_all = df_all[df_all[col_label].notnull()]
    period = (df_all.index > "2014-01-01")
    df_all = df_all[period]

    y = df_all[col_label] / df_all["open"] - 1

    # print distribution of y
    print("y<=-0.5:",sum(y<=-0.5))
    for i in range(-5,5):
        tmp1 = ((i*0.1)< y)
        tmp2 = (y <=((i+1)*0.1))
        if len(tmp1)==0 or len(tmp2)==0:
            tmp = [False]
        else:
            tmp = tmp1 & tmp2
        print("{0}<y<={1}:".format(i*0.1,(i+1)*0.1),
              sum(tmp))
    print("y>0.5",sum(y>0.5))

    threshold = 0.15
    y[y > threshold] = 1
    y[y <= threshold] = 0

    print("total positive", sum(y))

    features = df_all.columns.difference([col_label, "sh_code", "sz_code",
                                          "code"])
    X = df_all[features]

    condition = (X.index > "2018-01-01")
    X_train, y_train = X[~condition], y[~condition]
    X_test, y_test = X[condition], y[condition]

    print("test positive:", sum(y_test))

    y_prd_list = []
    colors = ["r", "b"]
    for clf, c in zip(reversed(clfs), colors):
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        y_prd_list.append([clf, t2 - t1, clf.predict_proba(X_test), c])

    for clf, t, y_prd_prob, c in y_prd_list:
        y_prd = np.where(y_prd_prob[:, 0] < 0.4, 1, 0)
        print(clf.classes_)
        print(y_prd.shape, sum(y_prd))

        print("accuracy", metrics.accuracy_score(y_test, y_prd))
        print("precison", metrics.precision_score(y_test, y_prd))
        print("recall", metrics.recall_score(y_test, y_prd))
        precision, recall, _ = metrics.precision_recall_curve(y_test, y_prd_prob[:, 1])

        plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.plot(recall, precision, color=c)
        print(clf, t)

    plt.show()


if __name__ == '__main__':
    main()
