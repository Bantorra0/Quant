import pandas as pd
import numpy as np

import data_prepare as dp
import db_operations as dbop
import constants as const


def simple_amt_weighted_MACD(df:pd.DataFrame,m=12,n=26,k=9):
    df = df.sort_index(ascending=True)
    ma_short = df["amt"].rolling(window=m).sum()*10/df["vol"].rolling(window=m).sum()
    ma_long = df["amt"].rolling(window=n).sum()*10/df["vol"].rolling(window=n).sum()
    dif = ma_short-ma_long
    dea = (dif*df["amt"]).rolling(window=k).sum()*10/df["amt"].rolling(window=k).sum()
    macd = dif-dea
    result = pd.concat([dif,dea,macd,macd/dea*100],axis=1)
    result.columns = ["dif","dea","macd","macd_pct"]
    return result


def MACD(df:pd.DataFrame):
    pass


def simple_KDJ(df:pd.DataFrame):
    pass


if __name__ == '__main__':
    cursor = dbop.connect_db("sqlite3").cursor()
    start = 20140101
    cursor.execute("select * from stock_day ")
    df = dbop.create_df(cursor, const.STOCK_DAY[const.TABLE],
                        start=start,
                        where_clause="code='600352.SH'"
                        ).set_index("date")
    # print(df)
    result = simple_amt_weighted_MACD(df)
    print(result)