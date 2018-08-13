import pandas as pd
import numpy as np
import tushare as ts
import pymysql
import datetime


def get_adj_data(df_daily:pd.DataFrame, df_adj_factor:pd.DataFrame):
    df = df_daily.merge(df_adj_factor[["trade_date","adj_factor"]],
                       on="trade_date")
    df["adj_open"] = df["open"]*df["adj_factor"]
    print(df.columns, df.shape, df_daily.shape, df_adj_factor.shape)
    return df


def _date(date:str):
    return date[:4]+"-"+date[4:6]+"-"+date[6:8]


def main():
    # 设置tushare pro的token并获取连接
    ts.set_token('ca7a0727b75dce94ad988adf953673340308f01bacf1a101d23f15fc')
    pro = ts.pro_api()
    # 设定获取日线行情的初始日期和终止日期，其中终止日期设定为昨天。
    start_dt = '20100101'
    time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
    end_dt = time_temp.strftime('%Y%m%d')

    code = '000001.SZ'
    df_daily = pro.daily(ts_code=code,start_date=start_dt,
                         end_date=end_dt)
    df_adj_factor =pro.adj_factor(ts_code=code)
    df = get_adj_data(df_daily,df_adj_factor)
    df2 = ts.get_h_data(code,autype="hfq",start=_date(start_dt),end=_date(end_dt))
    print(sum(df["adj_open"]!=df2["open"]))





if __name__ == '__main__':
    main()