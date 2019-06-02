import pandas as pd

import collect
import data_process as dp
import db_operations as dbop
from constants import *

if __name__ == '__main__':
    idx = pd.IndexSlice
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20000101
    df = dbop.create_df(cursor, INDEX_DAY[TABLE], start=start,
                        # where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                        # where_clause="code='600352.SH'",
                        )
    index_pool = collect.get_index_pool()
    df1 = dp.prepare_index_d(df)
    result1_10 = pd.DataFrame(index=index_pool["name"],
                              columns=[str(i) + "月" + period + "日" for i in
                                       range(1, 13) for period in
                                       ("1-10", "11-20", "20-30")])

    for mth in range(1, 13):
        for code in index_pool.index:
            for start, end in [(1, 10), (11, 20), (21, 30)]:
                result1_10.loc[
                    str(mth) + "月{0}-{1}日".format(start, end),
                    index_pool.loc[code, "name"]] = df1.loc[(df1.index.get_level_values(
                    "date").year >= 2010) & (df1.index.get_level_values(
                    "date").month == mth) & (df1.index.get_level_values(
                    "date").day >= start) & (df1.index.get_level_values(
                    "date").day <= end), :].loc[idx[code, :], "chg_pct"].mean()

    result1_10["平均值"] = result1_10.mean(axis=1)
    result1_10.loc["最大值"] = result1_10.max(axis=0)
    result1_10.loc["最小值"] = result1_10.min(axis=0)
