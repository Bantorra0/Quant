import db_operations as dbop
import df_operations as dfop
import collect as clct
import pandas as pd
import numpy as np


def _check_int(arg):
    if type(arg)!= int:
        raise ValueError("{} is not a int".format(arg))


def _check_iterable(arg):
    if not hasattr(arg, "__iter__"):
        raise ValueError("{} is not iterable".format(arg))


def _make_iterable(arg):
    if type(arg)==str or not hasattr(arg,"__iter__"):
        return [arg]
    else:
        return arg


def _prefix(prefix, df:pd.DataFrame,copy=False):
    if copy:
        df = df.copy()
    df.columns = list(map(lambda col: str(prefix) + "_" + col, df.columns))
    return df


def _move(days, df:pd.DataFrame,cols):
    _check_int(days)
    cols = _make_iterable(cols)

    if days>0:
        pre = "p{}move".format(abs(days))
        df_move = df[cols].iloc[days:].copy()
        df_move.index = df.index[:-days]
    else:
        pre = "f{}move".format(abs(days))
        df_move = df[cols].iloc[:days].copy()
        df_move.index = df.index[-days:]

    return _prefix(pre,df_move)


def _rolling_max(days, df:pd.DataFrame, cols):
    _check_int(days)
    cols = _make_iterable(cols)

    df_rolling = df[cols].rolling(window=abs(days)).max()
    if days>0:
        cols_rolling = list(map(lambda s:"f"+str(abs(days))+"max_"+str(s),cols))
    else:
        cols_rolling = list(map(lambda s:"p"+str(abs(days))+"max_"+str(s),cols))
        df_rolling = df_rolling.dropna()
        df_rolling.index = df.index[:len(df_rolling)]

    df_rolling.columns = cols_rolling
    return df_rolling


def _rolling_min(days, df:pd.DataFrame, cols):
    _check_int(days)
    cols = _make_iterable(cols)

    df_rolling = df[cols].rolling(window=abs(days)).min()
    if days>0:
        cols_rolling = list(map(lambda s:"f"+str(abs(days))+"min_"+str(s),
                                cols))
    else:
        cols_rolling = list(map(lambda s:"p"+str(abs(days))+"min_"+str(s),
                                cols))
        df_rolling = df_rolling.dropna()
        df_rolling.index = df.index[:len(df_rolling)]

    df_rolling.columns = cols_rolling
    return df_rolling


def _rolling_mean(days, df:pd.DataFrame, cols):
    _check_int(days)
    cols = _make_iterable(cols)

    df_rolling = df[cols].rolling(window=abs(days)).mean()
    if days>0:
        cols_rolling = list(map(lambda s:"f"+str(abs(days))+"mean_"+str(s),cols))
    else:
        cols_rolling = list(map(lambda s:"p"+str(abs(days))+"mean_"+str(s),cols))
        df_rolling = df_rolling.dropna()
        df_rolling.index = df.index[:len(df_rolling)]

    df_rolling.columns = cols_rolling
    return df_rolling


def main():
    db_type = "sqlite3"
    #
    # init_table(STOCK_DAY.TABLE_NAME, db_type)
    # collect_stock_day(STOCK_DAY.pools(),db_type)
    #
    # init_table(INDEX_DAY.TABLE_NAME, db_type)
    # collect_index_day(INDEX_DAY.pools(), db_type)


    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    sql_select_all = "select * from {}"

    stock_day, index_day = clct.STOCK_DAY[clct.TABLE], clct.INDEX_DAY[clct.TABLE]

    cursor.execute(sql_select_all.format(stock_day))
    df_stck_d = pd.DataFrame(cursor.fetchall())
    df_stck_d.columns = dbop.cols_from_cur(cursor)

    cursor.execute(sql_select_all.format(index_day))
    df_idx_d = pd.DataFrame(cursor.fetchall())
    df_idx_d.columns = dbop.cols_from_cur(cursor)

    df_stck_d = df_stck_d.set_index(["date"]).sort_index(ascending=False)

    df_stck_list = []
    pred_period = 20
    cols_move = ["open", "high", "low", "close"]
    cols_roll = ["open", "close"]
    for _,df in df_stck_d.groupby("code"):

        df_label_max = _rolling_max(pred_period,df,"high")
        col_label = df_label_max.columns
        # df_label_min = _rolling_min(pred_period,df,"low")
        df_move_list = [_move(i,df, cols_move) for i in range(1,6)]
        df_rolling_list = [[_rolling_max(i,df,cols_move),
                        _rolling_min(i,df,cols_roll),
                        _rolling_mean(i,df,cols_roll)]
                       for i in [-5,-10,-20]]
        df_rolling_list = np.concatenate(df_rolling_list).tolist()
        tmp = pd.concat([df] + df_move_list + df_rolling_list + [df_label_max],
                      axis=1,sort=False)
        df_stck_list.append(tmp)
        # print(tmp.shape)
        # print(tmp[tmp[col_label[0]].isnull()])

    df_stck_d_all = pd.concat(df_stck_list,sort=False)
    print(df_stck_d_all)

    # print(df_stck_d_all[df_stck_d_all["f20max_high"].isnull()])


    df_idx_d = df_idx_d.set_index("date").sort_index(ascending=False)
    df_idx_list = [_prefix(name,
                           pd.concat([group]+[_move(i,group,cols_move)
                                          for i in [1,2,3]],axis=1,
                                     sort=False))
                   for name,group in df_idx_d.groupby("code")]

    df_idx_d = pd.concat(df_idx_list,axis=1,sort=False)

    print(df_idx_d)
    df_all = df_stck_d_all.join(df_idx_d)

    #
    # print(df_all.shape)
    # print(df_all.columns)

    import xgboost.sklearn as xgb

    clf = xgb.XGBClassifier(n_estimators=20,scale_pos_weight=10,max_depth=4)

    print(col_label)
    print(df_all.shape)
    # print(df_all[df_all[col_label[0]].isnull()])
    df_all = df_all[df_all[col_label[0]].notnull()]
    print(df_all.shape)
    period = (df_all.index>"2015-01-01")
    df_all = df_all[period]
    print(df_all.shape)
    print(df_all[col_label])
    y = df_all[col_label[0]]/df_all["open"]-1
    y[y>0.1]=1
    y[y<=0.1]=0
    print(y)
    features = df_all.columns.difference([col_label[0],"sh_code","sz_code",
                                          "code"])
    X = df_all[features]

    condition = (X.index>"2018-01-01")
    X_train,y_train = X[~condition],y[~condition]
    X_test,y_test=X[condition],y[condition]

    clf.fit(X_train,y_train)

    y_prd_prob = clf.predict_proba(X_test)
    print(y_prd_prob)
    y_prd = np.where(y_prd_prob[:,0]<0.55,1,0)
    print(y_prd.shape,sum(y_prd))

    import sklearn.metrics as metrics

    print(clf.classes_)
    print(metrics.accuracy_score(y_test,y_prd))
    print(metrics.precision_score(y_test,y_prd))
    print(metrics.recall_score(y_test,y_prd))
    print(len(metrics.precision_recall_curve(y_test,y_prd_prob[:,1])))
    x,y,_=metrics.precision_recall_curve(y_test,y_prd_prob[:,1])
    import matplotlib.pyplot as plt
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    main()