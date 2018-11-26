import datetime

import numpy as np
import pandas as pd
import tushare as ts

import data_cleaning as dc
import db_operations as dbop
import df_operations as dfop
from constants import TOKEN, STOCK_DAY, INDEX_DAY, TABLE, COLUMNS


def stck_pools():
    api = _init_api(TOKEN)
    hgt_stcks = api.stock_basic(is_hs="H")
    sgt_stcks = api.stock_basic(is_hs="S")

    stcks = unify_df_col_nm(pd.concat([hgt_stcks, sgt_stcks]))[
        ["code", "symbol"]]

    # print(stcks)
    # stck_amt = pd.DataFrame(index=["code"],columns=["avg_amt"])
    # for code in stcks:
    #     print("------{}------".format(code))
    #     tmp = unify_df_col_nm(api.daily(ts_code=code, start_date="20180701"))
    #     stck_amt[code]=tmp["amt"].mean()
    #
    # stcks_large_amt = set(stck_amt.sort_values(by="avg_amt",ascending=False).index[:100])

    # return stcks & stcks_large_amt

    symbols = ['601318', '600519', '000063', '000651', '000858', '002359',
               '002415', '600276', '002236', '000333', '600536', '600887',
               '300724', '603799', '601336', '600703', '300747', '600050',
               '000725', '000636', '300059', '000002', '000001', '002027',
               '600779']

    symbols = pd.DataFrame(np.array(symbols).reshape(-1, 1),
                           columns=["symbol"]).set_index("symbol")
    stcks = stcks.set_index("symbol")

    pools = stcks.join(symbols, how="inner")["code"]

    stcks_5g = ['600487.SH', '601138.SH', '002217.SZ', '600522.SH',
                '002913.SZ', '002402.SZ', '600345.SH', '300292.SZ',
                '300038.SZ', '300113.SZ', '300068.SZ', '002446.SZ',
                '000070.SZ', '300679.SZ', '002335.SZ', '000063.SZ']

    return set(pools) | set(stcks_5g)


def idx_pools():
    return ["sh", "sz", "hs300", "sz50", "cyb"]


def _init_api(token=TOKEN):
    # 设置tushare pro的token并获取连接
    ts.set_token(token)
    return ts.pro_api()


def init_table(table_name, db_type):
    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    configs = dbop.parse_config(path="database\\config\\{}".format(table_name))
    sql_drop = "drop table {}".format(table_name)
    sql_create = configs["create"]
    print(sql_create)
    try:
        cursor.execute(sql_drop)
    except Exception as e:
        pass
    cursor.execute(sql_create)
    dbop.close_db(conn)


def _get_col_names(cursor):
    return [desc[0] for desc in cursor.description]


def unify_col_names(columns: list):
    columns = list(columns)
    mapping = {"ts_code": "code", "trade_date": "date", "amount": "amt",
               "volume": "vol", "change": "p_change"}
    for col1, col2 in mapping.items():
        if col1 in columns:
            idx = columns.index(col1)
            columns[idx] = col2
    return columns


def unify_df_col_nm(df: pd.DataFrame, copy=False):
    if copy:
        df = df.copy()

    new_columns = unify_col_names(df.columns)
    df.columns = new_columns
    return df


def insert_to_db(row, db_type: str, table_name, columns):
    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()
    cursor.execute(dbop._sql_insert(db_type, table_name, columns),
                   list(row[columns]))
    conn.commit()


def download_index_day(pools: [str], db_type:str, update=False,
                       start ="2000-01-01"):
    download_failure = 0
    for i, code in enumerate(pools):
        try:
            # Set start to the newest date in table if update is true.
            if update:
                latest_date = dbop.get_latest_date(INDEX_DAY[TABLE],code,db_type)
                if latest_date:
                    start = latest_date

            print("start:",start)

            df = ts.get_k_data(code=code, start=start)

            # Unify column names.
            df.columns = unify_col_names(df.columns)
            print(df.shape)

            # Print progress.
            # 打印进度
            print('Seq: ' + str(i + 1) + ' of ' + str(
                len(pools)) + '   Code: ' + str(code))

        except Exception as err:
            download_failure+=1
            print(err)
            print('No DATA Code: ' + str(i))
            continue

        yield df

    print("-"*10,"\nDownload failure:{0}\n".format(download_failure))
    yield download_failure


def download_stock_day(pools: [str], db_type:str, update=False,
                       start="2000-01-01"):
    pro = _init_api(TOKEN)
    download_failure = 0
    for i, code in enumerate(pools):
        try:
            # Set start to the newest date in table if update is true.
            if update:
                latest_date = dbop.get_latest_date(STOCK_DAY[TABLE],code,db_type)
                if latest_date:
                    start = latest_date

            print("start:",start)

            # Download daily data and adj_factor(复权因子).
            # pro.daily accepts start_date with format "YYYYmmdd".
            start = str(start).replace("-", "")
            daily = pro.daily(ts_code=code, start_date=start)
            adj_factor = pro.adj_factor(ts_code=code)
            if start:
                adj_factor = adj_factor[adj_factor["trade_date"] >= start]
                print("adj:",adj_factor.shape)
            # Combine both into one dataframe.
            df = dfop.natural_join(daily, adj_factor, how="outer")

            # Unify column names.
            df = unify_df_col_nm(df)

            # Unify date format from "YYYYmmdd" to "YYYY-mm-dd".
            df["date"] = df["date"].apply(
                lambda d:datetime.datetime.strptime(d,"%Y%m%d").strftime(
                    '%Y-%m-%d'))
            print(df.shape)

            # Print progress.
            # 打印进度。
            print('Seq: ' + str(i + 1) + ' of ' + str(
                len(pools)) + '   Code: ' + str(code)+"\n")

        except Exception as err:
            download_failure += 1
            print(err)
            print('No DATA Code: ' + str(i))
            continue

        yield df

    print("-"*10,"\nDownload failure:{0}\n".format(download_failure))
    yield download_failure


def collect_stock_day(pools: [str], db_type: str, update=False,
                      start="2000-01-01"):
    conn = dbop.connect_db(db_type)
    download_failure,write_failure=0,0
    for df_single_stock_day in download_stock_day(pools=pools,db_type=db_type,
                                      update=update, start=start):
        if type(df_single_stock_day)!=int:
            conn,failure = dbop.write2db(df_single_stock_day,table=STOCK_DAY[
                TABLE],
                      cols=STOCK_DAY[COLUMNS],conn=conn, close=False)
            write_failure +=failure
        else:
            download_failure = df_single_stock_day
        print()
    dbop.close_db(conn)
    return download_failure,write_failure


def collect_index_day(pools: [str], db_type: str, update=False,
                      start="2000-01-01"):
    conn = dbop.connect_db(db_type)
    download_failure,write_failure = 0,0
    for df_single_index_day in download_index_day(pools=pools,db_type=db_type,
                                      update=update, start=start):
        if type(df_single_index_day)!=int:
            conn, failure = dbop.write2db(df_single_index_day,
                                            table=INDEX_DAY[TABLE],
                      cols=INDEX_DAY[COLUMNS],conn=conn, close=False)
            write_failure += failure
        else:
            download_failure = df_single_index_day
        print()
    dbop.close_db(conn)
    return download_failure,write_failure


def update():
    db_type = "sqlite3"

    # # init_table(INDEX_DAY[TABLE], db_type)
    # # init_table(STOCK_DAY[TABLE], db_type)

    index_pools = idx_pools()
    # stock_pools = stck_pools()
    stock_pools = ['600050.SH', '600276.SH', '600519.SH', '600536.SH',
                   '600703.SH', '600779.SH', '600887.SH', '601318.SH',
                   '601336.SH', '603799.SH', '000001.SZ', '000002.SZ',
                   '000063.SZ', '000636.SZ', '000651.SZ', '000858.SZ',
                   '002027.SZ', '002236.SZ', '002359.SZ', '002415.SZ',
                   '300059.SZ', '600345.SH', '300068.SZ', '300038.SZ',
                   '300292.SZ', '300113.SZ', '002446.SZ', '002402.SZ',
                   '300679.SZ', '002335.SZ', '000070.SZ', '002913.SZ',
                   '601138.SH', '002217.SZ', '600522.SH', '600401.SH',
                   '600487.SH', '600567.SH', '002068.SZ', '000488.SZ',
                   '600392.SH', '600966.SH', '000725.SZ', '600549.SH',
                   '000333.SZ', '300700.SZ', '000338.SZ', '002099.SZ',
                   '600023.SH', '000581.SZ', '000539.SZ']


    download_failure = 1
    write_failure = 0
    while download_failure>0 or write_failure>0:
        print("Indexes:", len(index_pools))
        download_failure1,write_failure1 = collect_index_day(index_pools,
                                                           db_type,
                                              update=True)
        print("Stocks:",len(stock_pools))
        download_failure2, write_failure2 = collect_stock_day(stock_pools,
                                                              db_type,
                                             update=False,start="2018-11-01")
        download_failure = download_failure1+download_failure2
        write_failure = write_failure1+write_failure2
    dc.fillna_stock_day(db_type=db_type)


if __name__ == '__main__':
    update()
