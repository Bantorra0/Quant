from collecting import _init_api,TOKEN
from df_operations import natural_outer_join
import tushare as ts


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
    print(ts.get_k_data(code="sh"))



