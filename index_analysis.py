import pandas as pd
import numpy as np

import collect
import data_process as dp
import db_operations as dbop
from constants import *


def argmax(df:pd.DataFrame,axis=0):
    if axis==0:
        labels = df.index
    elif axis==1:
        labels = df.columns
    else:
        raise ValueError("axis={} is illegal!".format(axis))

    return labels[np.argmax(df.values,axis=axis)]


def argmin(df: pd.DataFrame, axis=0):
    if axis == 0:
        labels = df.index
    elif axis == 1:
        labels = df.columns
    else:
        raise ValueError("axis={} is illegal!".format(axis))

    return labels[np.argmin(df.values, axis=axis)]


def period_agg(df:pd.DataFrame, op=np.mean, start=2000):
    df_result = pd.DataFrame(columns=index_pool["name"])
    df = df.loc[df.index.get_level_values("date").year >= start,:]

    for mth in range(1, 13):
        mask0 = (df.index.get_level_values("date").month == mth)
        for code in index_pool.index:
            mask1 = (df.index.get_level_values("code") == code)
            for start, end in [(1, 10), (11, 20), (21, 30)]:
                mask2 = (df.index.get_level_values("date").day >= start)\
                        & (df.index.get_level_values("date").day <= end)
                mask = mask0 & mask1 & mask2
                index_label = str(mth) + "月{0}-{1}日".format(start, end)
                col_label = index_pool.loc[code, "name"]
                df_result.loc[index_label,col_label] = op(df.loc[mask,"chg_pct"])

    return df_result


def week_agg(df:pd.DataFrame, op=np.mean, start=2000):
    df_result = pd.DataFrame(columns=index_pool["name"])
    df = df.loc[df.index.get_level_values("date").year >= start,:]

    range_ = range(1, 54)
    for week in range_:
        mask0 = (df.index.get_level_values("date").week == week)
        for code in index_pool.index:
            mask1 = (df.index.get_level_values("code") == code)
            mask = mask0 & mask1
            index_label = "第{}周".format(week)
            col_label = index_pool.loc[code, "name"]
            df_result.loc[index_label,col_label] = op(df.loc[mask,"chg_pct"])

    return df_result


def dayofweek_agg(df:pd.DataFrame, op=np.mean, start=2000):
    df_result = pd.DataFrame(columns=index_pool["name"])
    df = df.loc[df.index.get_level_values("date").year >= start,:]
    range_ = range(5)

    for dayofweek in range_:
        mask0 = (df.index.get_level_values("date").dayofweek == dayofweek)
        dayofweek_names = {i:"周"+char for i,char in zip(range_,list("一二三四五"))}
        for code in index_pool.index:
            mask1 = (df.index.get_level_values("code") == code)
            mask = mask0 & mask1
            index_label = dayofweek_names[dayofweek]
            col_label = index_pool.loc[code, "name"]
            df_result.loc[index_label,col_label] = op(df.loc[mask,"chg_pct"])

    return df_result


def df_statistics(df:pd.DataFrame,row_ops=None,col_ops=None,ops=None):
    if ops is not None:
        row_ops = ops
        col_ops = ops

    result = df.copy()
    index = df.index
    columns = df.columns

    new_cols = pd.DataFrame(index=index)
    for col,op,kwargs in col_ops:
        new_cols[col] = op(df,axis=1,**kwargs)

    new_rows = pd.DataFrame(columns=columns)
    for row_index,op,kwargs in row_ops:
        new_rows.loc[row_index] = op(df,axis=0,**kwargs)

    result = pd.concat([result,new_cols],axis=1,sort=False)
    result = pd.concat([result,new_rows],axis=0,sort=False)

    # row_mean = df_result.mean(axis=1)
    # row_max = df_result.max(axis=1)
    # row_argmax = df_result.columns[df_result.values.argmax(axis=1)]
    # row_min = df_result.min(axis=1)
    # row_argmin = df_result.columns[df_result.values.argmin(axis=1)]
    #
    # col_mean = df_result.mean(axis=0)
    # col_max = df_result.max(axis=0)
    # col_argmax = df_result.index[df_result.values.argmax(axis=0)]
    # col_min = df_result.min(axis=0)
    # col_argmin = df_result.index[df_result.values.argmin(axis=0)]
    #
    # df_result["平均值"] = row_mean
    # df_result["最大值"] = row_max
    # df_result["最大值对应指数"] = row_argmax
    # df_result["最小值"] = row_min
    # df_result["最小值对应指数"] = row_argmin
    #
    # df_result.loc["平均值",columns] = col_mean
    # df_result.loc["最大值",columns] = col_max
    # df_result.loc["最大值对应时间",columns] = col_argmax
    # df_result.loc["最小值",columns] = col_min
    # df_result.loc["最小值对应时间",columns] = col_argmin

    print(result.to_string().replace("NaN","").replace("nan",""))
    return result


if __name__ == '__main__':
    idx = pd.IndexSlice
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20000101
    df = dbop.create_df(cursor, INDEX_DAY[TABLE], start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600352.SH'",
                        )
    index_pool = collect.get_index_pool().set_index("code")
    df = dp.prepare_index_d(df)
    df.sort_index(inplace=True)
    df["chg_pct"] = (df["close"]/df["close"].groupby(level="code").shift(1)-1)*100
    # result1_10 = pd.DataFrame(columns=index_pool["name"])
    # columns = [str(i) + "月" + period + "日" for i in
    #            range(1, 13) for period in
    #            ("1-10", "11-20", "20-30")]

    ops = [("mean",np.mean,{}),("median",np.median,{}),
           ("max",np.max,{}),("argmax",argmax,{}),
           ("min",np.min,{}),("argmin",argmin,{}),
           ("std",np.std,{})
           ]

    result = df_statistics(dayofweek_agg(df, op=np.mean,start=2010), ops=ops)
    result = df_statistics(dayofweek_agg(df, op=np.nanmedian,start=2010), ops=ops)
    result = df_statistics(dayofweek_agg(df, op=np.std,start=2010), ops=ops)


