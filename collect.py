import datetime

import numpy as np
import pandas as pd
import tushare as ts

from constants import TOKEN, STOCK_DAY, INDEX_DAY,TABLE,COLUMNS
from db_operations import connect_db, _parse_config, close_db
from df_operations import natural_outer_join



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

    pools = list(stcks.join(symbols, how="inner")["code"])

    stcks_5g = ['600487.SH', '601138.SH', '002217.SZ', '600522.SH',
                '002913.SZ', '002402.SZ', '600345.SH', '300292.SZ',
                '300038.SZ', '300113.SZ', '300068.SZ', '002446.SZ',
                '000070.SZ', '300679.SZ', '002335.SZ', '000063.SZ']

    return set(pools) | set(stcks_5g)


def idx_pools():
    return ["sh", "sz", "hs300", "sz50", "cyb"]


def _sql_insert(db: str, table_name: str, cols: [str]):
    placeholders = {"mysql": "%s", "sqlite3": "?"}
    if db in placeholders:
        sql_insert = "INSERT INTO {} ".format(table_name) + "({})".format(
            ",".join(cols)) + "VALUES ({})"
        return sql_insert.format(",".join([placeholders[db]] * len(cols)))
    else:
        raise ValueError("{} not supported".format(db))


def _init_api(token=TOKEN):
    # 设置tushare pro的token并获取连接
    ts.set_token(token)
    return ts.pro_api()


def init_table(table_name, db_type):
    conn = connect_db(db_type)
    cursor = conn.cursor()

    configs = _parse_config(path="database\\config\\{}".format(table_name))
    sql_drop = "drop table {}".format(table_name)
    sql_create = configs["create"]
    print(sql_create)
    try:
        cursor.execute(sql_drop)
    except Exception as e:
        pass
    cursor.execute(sql_create)
    close_db(conn)


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
    conn = connect_db(db_type)
    cursor = conn.cursor()
    cursor.execute(_sql_insert(db_type, table_name, columns),
                   list(row[columns]))
    conn.commit()


def collect_index_day(pools: [str], db_type: str, update=False):
    conn = connect_db(db_type)
    cursor = conn.cursor()
    for i, code in enumerate(pools):
        try:
            start = "2000-01-01"
            if update:
                cursor.execute("select date from {0} where code='{1}'".format(
                    INDEX_DAY[TABLE],code))
                rs = cursor.fetchall()
                print(rs)
                if len(rs)>0:
                    start = sorted(rs,reverse=True)[0][0]
                    cursor.execute(("delete from {0} where code='{1}' "
                                   "and date='{2}'").format(INDEX_DAY[TABLE], code, start))

            print("start:",start)
            df = ts.get_k_data(code=code, start=start)
            # 打印进度
            print('Seq: ' + str(i + 1) + ' of ' + str(
                len(pools)) + '   Code: ' + str(code))
            df.columns = unify_col_names(df.columns)
            print(df.shape)

        except Exception as err:
            print(err)
            print('No DATA Code: ' + str(i))
            continue

        for _, row in df.iterrows():
            try:
                cursor.execute(
                    _sql_insert(db_type, INDEX_DAY[TABLE], INDEX_DAY[COLUMNS]),
                    tuple(row[list(INDEX_DAY[COLUMNS])]))
                conn.commit()
            except Exception as err:
                print(err)

                continue
    close_db(conn)


def collect_stock_day(pools: [str], db_type: str, update=False):
    pro = _init_api(TOKEN)
    conn = connect_db(db_type)
    cursor = conn.cursor()
    for i, code in enumerate(pools):
        try:
            start = "20000101"
            if update:
                cursor.execute("select date from {0} where code='{1}'".format(
                    STOCK_DAY[TABLE],code))
                rs = cursor.fetchall()
                if len(rs)>0:
                    start = sorted(rs,reverse=True)[0][0]
                    cursor.execute(("delete from {0} where code='{1}' "
                                   "and date='{2}'").format(STOCK_DAY[TABLE],
                                                            code, start))
                    start = str(start).replace("-","")

            print("start:",start)
            daily = pro.daily(ts_code=code, start_date=start)
            adj_factor = pro.adj_factor(ts_code=code)

            if update:
                adj_factor = adj_factor[adj_factor["trade_date"]>=start]
                print("adj:",adj_factor.shape)

            df = natural_outer_join(daily, adj_factor)
            # 打印进度
            print('Seq: ' + str(i + 1) + ' of ' + str(
                len(pools)) + '   Code: ' + str(code))
            df = unify_df_col_nm(df)
            print(df.columns)

        except Exception as err:
            print(err)
            print('No DATA Code: ' + str(i))
            continue

        for _, row in df.iterrows():
            try:
                row["date"] = (datetime.datetime.strptime(row["date"],
                                                          "%Y%m%d")).strftime(
                    '%Y-%m-%d')

                cursor.execute(
                    _sql_insert(db_type, STOCK_DAY[TABLE], STOCK_DAY[COLUMNS]),
                    tuple(row[list(STOCK_DAY[COLUMNS])]))
                conn.commit()

            except Exception as err:
                print("error:",err)
                continue
    close_db(conn)


def main():
    db_type = "sqlite3"

    # init_table(STOCK_DAY[TABLE], db_type)
    print(len(stck_pools()))
    collect_stock_day(stck_pools(), db_type, update=True)

    # init_table(INDEX_DAY[TABLE], db_type)
    collect_index_day(idx_pools(), db_type, update=True)

    # conn = connect_db(db_type)
    # cursor = conn.cursor()
    # # print(cursor.execute("select * from stock_day").fetchmany(50))
    # print(cursor.fetchmany(100))
    #
    # cursor.execute("select * from index_day")
    # print(cursor.fetchmany(100))
    # # print(cursor.execute("select * from stock_day").fetchmany(100))

    # print(stck_pools())


if __name__ == '__main__':
    main()
