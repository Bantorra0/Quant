import tushare as ts

import collect



if __name__ == '__main__':
    print(ts.__version__)
    api = collect._init_api()
    # df = ts.pro_bar(pro_api=api, ts_code='000001.SH', asset='I',
    #                 start_date='20180601', end_date='20180701')
    # print(df.shape)
    # print(df.columns)
    # print(df.iloc[:20])

    df_sb_l = api.stock_basic(exchange='', list_status='L')
    print(df_sb_l.shape)
    print(df_sb_l.head())
    df_sb_p = api.stock_basic(exchange='', list_status='P')
    print(df_sb_p.shape)
    print(df_sb_p.head())

