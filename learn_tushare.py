import tushare as ts

import collect



if __name__ == '__main__':
    print(ts.__version__)
    api = collect._init_api()
    df = ts.pro_bar(pro_api=api, ts_code='000001.SH', asset='I',
                    start_date='20180601', end_date='20180701')
    print(df.shape)
    print(df.columns)
    print(df.iloc[:20])
