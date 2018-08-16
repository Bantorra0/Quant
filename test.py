from collect import _init_api,TOKEN
from df_operations import natural_outer_join
import tushare as ts
import pandas as pd
import datetime

if __name__ == '__main__':
    api = _init_api(TOKEN)
    # df = api.daily(ts_code="002099.SZ", start_date="20180701")
    # print(df.iloc[:10,:])
    #
    #
    # df2 = api.adj_factor(ts_code="002099.SZ")
    # print(df2.iloc[:10,:])
    # print(natural_outer_join(df,df2))
    # print(api.daily(ts_code="sh",index=True))
    # print(api.adj_factor(ts_code="399001.SZ"))

    # df1 = api.daily(ts_code="000001.SZ")
    # df1.to_csv("daily.csv")
    # df2 = ts.get_k_data(code="000001",start="1991-01-01")
    # df2.to_csv("k_daily.csv")

    df1 = pd.read_csv("daily.csv",index_col=0)
    df2= pd.read_csv("k_daily.csv",index_col=0)

    f = lambda s:datetime.datetime.strptime(str(s),"%Y%m%d").strftime("%Y-%m-%d")
    df1["date"] = df1["trade_date"].apply(f)

    df1 = df1.set_index(keys="date")
    df2 = df2.set_index(keys="date")


    print(df1.loc[df1.index.difference(df2.index)])
    print(df2.loc[df2.index.difference(df1.index)])


    # import time
    # t1 = time.time()
    # df1 = ts.get_k_data(code="000581", start="1991-01-01")
    # t2= time.time()
    # print(t2-t1)
    # print(df1)
    # print()
    #
    # t3= time.time()
    # df2 = api.daily(ts_code="000581.SZ")
    # t4 = time.time()
    # print(t4-t3)
    # print(df2)
    # print()


