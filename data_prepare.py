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

    df_stck_d = df_stck_d.set_index(["date","code"])
    df_idx_d = df_idx_d.set_index("date")

    df = df_stck_d.copy()

    pred_period = 20
    df_label_max = _rolling_max(pred_period,df,"high")
    print(type(df_label_max))
    col_label = df_label_max.columns
    # df_label_min = _rolling_min(pred_period,df,"low")

    cols_move = ["open","high","low","close"]
    df_move_list = [_move(i,df, cols_move) for i in range(1,6)]

    cols_roll = ["open","close"]
    df_rolling_list = [[_rolling_max(i,df,cols_move),
                        _rolling_min(i,df,cols_roll),
                        _rolling_mean(i,df,cols_roll)]
                       for i in [-5,-10,-20]]
    df_rolling_list = np.concatenate(df_rolling_list).tolist()

    df_idx_list = [_prefix(name,group) for name,group in df_idx_d.groupby(
        "code")]

    idx_dc_list = [df]+df_move_list + df_rolling_list + [df_label_max]
    idx_d_list = df_idx_list

    df_idx_dc = pd.concat(idx_dc_list,axis=1,sort=False)
    df_idx_d = pd.concat(idx_d_list,axis=1,sort=False)

    df_idx_dc.reset_index(level=1,inplace=True)

    df_train = df_idx_dc.join(df_idx_d)

    columns = np.concatenate([list(df.columns) for df in idx_dc_list+idx_d_list])

    print(df_train.shape)

    df_train = df_train[columns]
    print(df_train.columns)
    #
    # print(df_train.shape)
    # print(df_train.columns)


if __name__ == '__main__':
    main()