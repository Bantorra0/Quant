import datetime
import time
import multiprocessing as mp
import pickle

import numpy as np
import pandas as pd
import tushare as ts

import constants as const
import data_cleaning as dc
import db_operations as dbop
import df_operations as dfop


def stck_pools():
    api = _init_api(const.TOKEN)
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


def _init_api(token=const.TOKEN):
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


def download_single_index_day(code, db_type: str, update=False,
                              start="2000-01-01", end=None, verbose=0):
    try:
        # Set start to the newest date in table if update is true.
        if update:
            latest_date = dbop.get_latest_date(const.INDEX_DAY[const.TABLE], code, db_type)
            if latest_date:
                start = latest_date
        if verbose > -1:
            print("start:", start)

        df = ts.get_k_data(code=code, start=start, end=end)

    except Exception as err:
        print(err)
        print('No DATA Code:', code)
        return False

    else:
        # df["date"] = df["date"].astype(str)
        # Unify column names.
        df.columns = unify_col_names(df.columns)
        if verbose > -1:
            print(df.shape)
        return df


def download_single_stock_day(code, db_type: str, update=False,
                              start="2000-01-01", end=None, verbose=0):
    pro = _init_api(const.TOKEN)
    try:
        # Set start to the newest date in table if update is true.
        if update:
            latest_date = dbop.get_latest_date(const.STOCK_DAY[const.TABLE], code, db_type)
            if latest_date:
                start = datetime.datetime.strptime(latest_date, "%Y-%m-%d") - datetime.timedelta(days=5)
                start = start.strftime('%Y-%m-%d')

        if verbose > -1:
            print("start:", start)

        # Download daily data and adj_factor(复权因子).
        # pro.daily accepts start_date with format "YYYYmmdd".
        start = str(start).replace("-", "")
        if end:
            end = str(end).replace("-", "")

        kwargs = {"ts_code":code, "start_date":start,
                            "end_date":end}
        daily = pro.daily(**kwargs)
        adj_factor = pro.adj_factor(**kwargs)

    except Exception as err:
        print(err)
        print('No DATA Code:', code)
        return False

    else:
        # daily["trade_date"] = daily["trade_date"].astype(str)
        # adj_factor["trade_date"] = adj_factor["trade_date"].astype(str)
        if verbose > -1:
            print(daily["trade_date"].max(), str(adj_factor["trade_date"].max()))

        if len(daily)>0 and len(adj_factor)>0:
            # print("daily.shape:",daily.shape, "adj_factor.shape:",adj_factor.shape)
            end = min(daily["trade_date"].max(), adj_factor["trade_date"].max())
            start = max(daily["trade_date"].min(), adj_factor["trade_date"].min())

            daily = daily[(daily["trade_date"] <= end) & (daily["trade_date"] >= start)]
            adj_factor = adj_factor[(adj_factor["trade_date"] <= end) & (adj_factor["trade_date"] >= start)]

        # Join two dataframes.
        df = dfop.natural_join(daily, adj_factor, how="outer")

        # Unify column names.
        df = unify_df_col_nm(df)

        # Unify date format from "YYYYmmdd" to "YYYY-mm-dd".
        df["date"] = df["date"].apply(
            lambda d: datetime.datetime.strptime(d, "%Y%m%d").strftime(
                '%Y-%m-%d'))

        if verbose > -1:
            print(df.shape)

        return df


def download_stock_basic(db_type: str):
    pro = _init_api(const.TOKEN)
    download_failure = 0
    status_list = ["L", "D", "P"]  # L上市，D退市，P暂停上市。
    fields = ['ts_code', 'symbol', 'name', 'area', 'industry', 'fullname',
              'enname', 'market', 'exchange', 'curr_type', 'list_status',
              'list_date', 'delist_date', 'is_hs']
    df_list = []
    try:
        for status in status_list:
            df_list.append(
                pro.stock_basic(list_status=status, fields=",".join(fields)))
            print(df_list[-1].shape)

        df = pd.concat(df_list, sort=False, ignore_index=True)
        df.columns = unify_col_names(df.columns)
        print(df.shape)
        print(df.columns)
        yield df

    except Exception as err:
        download_failure = 1
        print(err)

    print("-" * 10, "\nDownload failure:{0}\n".format(download_failure))
    yield download_failure


def collect_single_stock_day(code, db_type: str, update=False,
                             start="2000-01-01", verbose=0, conn=None, close_db=False):
    if conn is None:
        conn = dbop.connect_db(db_type)
    download_failure, write_failure = 0, 0
    df_single_stock_day = download_single_stock_day(code=code, db_type=db_type,
                                                    update=update, start=start,
                                                    verbose=verbose)
    if type(df_single_stock_day)==pd.DataFrame:
        conn, write_failure = dbop.write2db(df_single_stock_day,
                                            table=const.STOCK_DAY[const.TABLE],
                                            cols=const.STOCK_DAY[const.COLUMNS], conn=conn,
                                            close=False)
    elif not df_single_stock_day:
        # df_single_stock_day==False, download fails.
        download_failure = 1
    else:
        raise ValueError

    if close_db:
        dbop.close_db(conn)
    return download_failure, write_failure


def collect_single_index_day(code:str, db_type: str, update=False,
                             start="2000-01-01", verbose=0,conn=None):
    if conn is None:
        conn = dbop.connect_db(db_type)
    download_failure, write_failure = 0, 0
    df_single_index_day = download_single_index_day(code=code, db_type=db_type,
                                                    update=update, start=start,
                                                    verbose=verbose)
    if type(df_single_index_day)!=bool:
        conn, write_failure = dbop.write2db(df_single_index_day,
                                            table=const.INDEX_DAY[const.TABLE],
                                            cols=const.INDEX_DAY[const.COLUMNS], conn=conn,
                                            close=False)
    elif not df_single_index_day:
        download_failure = 1
    else:
        raise ValueError

    return download_failure, write_failure,conn


def collect_stock_basic(db_type: str, update=False):
    conn = dbop.connect_db(db_type)
    download_failure, write_failure = 0, 0
    for df_single_stock_basic in download_stock_basic(db_type=db_type):
        if type(df_single_stock_basic) == pd.DataFrame:
            conn, failure = dbop.write2db(df_single_stock_basic,
                                          table=const.STOCK_BASIC[const.TABLE],
                                          cols=const.STOCK_BASIC[const.COLUMNS],
                                          conn=conn, close=False)
            write_failure += failure
        elif type(df_single_stock_basic)==int and df_single_stock_basic>0:
            download_failure = df_single_stock_basic
        else:
            raise ValueError
        time.sleep(1)
    dbop.close_db(conn)
    return download_failure, write_failure


def update(db_type="sqlite3"):
    # # init_table(INDEX_DAY[TABLE], db_type)
    # # init_table(STOCK_DAY[TABLE], db_type)

    # stock_pool = stck_pools()
    # stock_pool = ['600050.SH', '600276.SH', '600519.SH', '600536.SH',
    #                '600703.SH', '600779.SH', '600887.SH', '601318.SH',
    #                '601336.SH', '603799.SH', '000001.SZ', '000002.SZ',
    #                '000063.SZ', '000636.SZ', '000651.SZ', '000858.SZ',
    #                '002027.SZ', '002236.SZ', '002359.SZ', '002415.SZ',
    #                '300059.SZ', '600345.SH', '300068.SZ', '300038.SZ',
    #                '300292.SZ', '300113.SZ', '002446.SZ', '002402.SZ',
    #                '300679.SZ', '002335.SZ', '000070.SZ', '002913.SZ',
    #                '601138.SH', '002217.SZ', '600522.SH', '600401.SH',
    #                '600487.SH', '600567.SH', '002068.SZ', '000488.SZ',
    #                '600392.SH', '600966.SH', '000725.SZ', '600549.SH',
    #                '000333.SZ', '300700.SZ', '000338.SZ', '002099.SZ',
    #                '600023.SH', '000581.SZ', '000539.SZ']

    index_pool = dbop.get_all_indexes()
    stock_pool = dbop.get_all_stocks()

    update_indexes(index_pool, db_type)
    update_stocks(stock_pool, db_type)

    dc.fillna_stock_day(db_type=db_type)


def update_indexes(index_pool, db_type="sqlite3", verbose=0, print_freq=1):
    print("Indexes:", len(index_pool))
    t0 = time.time()
    conn = dbop.connect_db(db_type)
    for i, index in enumerate(index_pool):
        if i % print_freq == 0:
            print('Seq:', str(i + 1), 'of', str(len(index_pool)), '  Code:', str(index))
        download_failure = 1
        write_failure = 0
        while download_failure > 0 or write_failure > 0:
            download_failure, write_failure, conn = collect_single_index_day(
                index, db_type, update=True, verbose=verbose, conn=conn)

            # Sleep to make sure each iteration take 0.3s,
            # because the server has a limit of 200 api connections per min.
            t1 = time.time()
            if t1 - t0 < 0.3:
                time.sleep(0.3 - (t1 - t0))
            t0 = t1
    dbop.close_db(conn)


def update_stocks(stock_pool, db_type="sqlite3", verbose=0, print_freq=1):
    print("Stocks:", len(stock_pool))
    conn=dbop.connect_db(db_type)

    pool = mp.Pool(processes=1)

    # Use manager.dict() or manager.list() to share objects between processes.
    # Return mutable objects (refs) may cause error because memory is not shared between process..
    # manager = mp.Manager()
    # d = manager.dict()

    start_time = time.time()
    for i, stock in enumerate(stock_pool):
        if i % print_freq == 0:
            print('Seq:', str(i + 1), 'of', str(len(stock_pool)), '  Code:', str(stock))
        download_failure = 1
        write_failure = 0
        while download_failure > 0 or write_failure > 0:
            kwargs = {"code": stock, "db_type": db_type,
                      "update": True, "verbose": verbose,
                      "conn":None, "close_db":True,
                      }

            # Use pickle to send and receive objects in mp,
            # which may raise error in case of unpicklable objects, e.g. sqlite3.connector.
            # That's why conn is not passed, returned and reused.
            res = pool.apply_async(func=collect_single_stock_day,kwds=kwargs)
            try:
                t0 =time.time()

                download_failure, write_failure = res.get(
                    timeout=const.TIMEOUT)
                t1 = time.time()
                if t1-t0<0.3:
                    time.sleep(0.3-(t1-t0))
            except mp.TimeoutError as err:
                print("Timeout: {}s,".format(const.TIMEOUT), type(err))
                download_failure=1
                pool.terminate()
                pool = mp.Pool(processes=1)
                continue

    end_time = time.time()

    print("Total collecting time: {:.1f}s".format(end_time-start_time))

    if conn:
        dbop.close_db(conn)


def download_single_stock_daily_basic(code, db_type: str, update=False,
                              start="2000-01-01", end=None, verbose=0):
    pro = _init_api(const.TOKEN)
    try:
        # Set start to the newest date in table if update is true.
        if update:
            latest_date = dbop.get_latest_date(const.STOCK_DAILY_BASIC[const.TABLE], code, db_type)
            if latest_date:
                start = datetime.datetime.strptime(latest_date, "%Y-%m-%d") - datetime.timedelta(days=5)
                start = start.strftime('%Y-%m-%d')

        if verbose > -1:
            print("start:", start)

        # Download daily_basic data and adj_factor(复权因子).
        # pro.daily_basic accepts start_date with format "YYYYmmdd".
        start = str(start).replace("-", "")
        if end:
            end = str(end).replace("-", "")

        kwargs = {"ts_code":code, "start_date":start, "end_date":end}
        daily_basic = pro.daily_basic(**kwargs)

    except Exception as err:
        print(err)
        print('No DATA Code:', code)
        return False

    else:
        # Unify column names.
        df = unify_df_col_nm(daily_basic)

        # Unify date format from "YYYYmmdd" to "YYYY-mm-dd".
        df["date"] = df["date"].apply(
            lambda d: datetime.datetime.strptime(d, "%Y%m%d").strftime(
                '%Y-%m-%d'))

        if verbose > -1:
            print(df["date"].min(), daily_basic["date"].max())
            print(df.shape)
        return df


def collect_single_stock_daily_basic(code, db_type: str, update=False,
                             start="2000-01-01", verbose=0, conn=None, close_db=False):
    if conn is None:
        conn = dbop.connect_db(db_type)
    download_failure, write_failure = 0, 0
    df_single_stock_daily_basic = download_single_stock_daily_basic(code=code, db_type=db_type,
                                                    update=update, start=start,
                                                    verbose=verbose)
    if type(df_single_stock_daily_basic)==pd.DataFrame:
        conn, write_failure = dbop.write2db(df_single_stock_daily_basic,
                                            table=const.STOCK_DAY[const.TABLE],
                                            cols=const.STOCK_DAY[const.COLUMNS], conn=conn,
                                            close=False)
    elif not df_single_stock_daily_basic:
        # df_single_stock_daily_basic==False, download fails.
        download_failure = 1

    if close_db:
        dbop.close_db(conn)
    return download_failure, write_failure


def update_stock_daily_basic(stock_pool, db_type="sqlite3", verbose=0, print_freq=1):
    print("Stocks:", len(stock_pool))
    conn=dbop.connect_db(db_type)

    pool = mp.Pool(processes=1)

    # Use manager.dict() or manager.list() to share objects between processes.
    # Return mutable objects (refs) may cause error because memory is not shared between process..
    # manager = mp.Manager()
    # d = manager.dict()

    start_time = time.time()
    for i, stock in enumerate(stock_pool):
        if i % print_freq == 0:
            print('Seq:', str(i + 1), 'of', str(len(stock_pool)), '  Code:', str(stock))
        kwargs = {"code": stock, "db_type": db_type,
                  "update": True, "verbose": verbose,
                  "conn": None, "close_db": True,
                  }

        download_failure = 1
        write_failure = 0
        while download_failure > 0 or write_failure > 0:
            # Use pickle to send and receive objects in mp,
            # which may raise error in case of unpicklable objects, e.g. sqlite3.connector.
            # That's why conn is not passed, returned and reused.
            res = pool.apply_async(func=collect_single_stock_daily_basic,kwds=kwargs)
            try:
                t0 =time.time()

                download_failure, write_failure = res.get(
                    timeout=const.TIMEOUT)
                t1 = time.time()
                if t1-t0<0.3:
                    time.sleep(0.3-(t1-t0))
            except mp.TimeoutError as err:
                print("Timeout: {}s,".format(const.TIMEOUT), type(err))
                download_failure=1
                pool.terminate()
                pool = mp.Pool(processes=1)
                continue

    end_time = time.time()

    print("Total collecting time: {:.1f}s".format(end_time-start_time))

    if conn:
        dbop.close_db(conn)


def update_stock_basic(db_type="sqlite3", initialize=False):
    if initialize:
        init_table(const.STOCK_BASIC[const.TABLE], "sqlite3")

    download_failure = 1
    write_failure = 0
    while download_failure > 0 or write_failure > 0:
        download_failure, write_failure = collect_stock_basic(db_type,
                                                              update=True)

def update_stock_list(stock_pool=None, db_type="sqlite3",cursor=None):
    if stock_pool is None:
        if cursor is None:
            cursor = dbop.connect_db(db_type).cursor()
        stock_pool = dbop.get_all_stocks(db_type, cursor)
    stock_pool = set(stock_pool)
    print("Stocks in table stock_d:", len(stock_pool))
    with open(r"database\stock_d_stock_list","wb") as f:
        pickle.dump(set(stock_pool),f)


def get_stock_pool():
    with open(r"database\stock_d_stock_list","rb") as f:
        stock_pool = pickle.load(f)
    return stock_pool




if __name__ == '__main__':
    # p = mp.Process(target=collect_single_stock_day, kwargs=kwargs)
    # p.start()
    # time.sleep(0.1)
    # if p.is_alive():
    #     p.terminate()
    #     print("Kill")

    db_type = "sqlite3"

    # update_stock_basic()

    # index_pool = dbop.get_all_indexes()
    # update_indexes(index_pool,db_type)

    stock_pool = get_stock_pool()
    # update_stocks(stock_pool, db_type=db_type)

    # init_table(const.STOCK_DAILY_BASIC[const.TABLE],db_type=db_type)
    update_stock_daily_basic(stock_pool=stock_pool,db_type=db_type)

    # dc.fillna_stock_day(db_type=db_type,start="2000-01-01")




