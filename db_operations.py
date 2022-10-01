import sqlite3
import pandas as pd
import pymysql
from constants import STOCK_DAY,INDEX_DAY,TABLE,COLUMNS


def connect_db(db_type:str,timeout=10):
    if db_type== "mysql":
        return pymysql.connect(host='127.0.0.1', user='root', passwd='Bantorra',
                             db='quant', charset='utf8')
    elif db_type == "sqlite3":
        return sqlite3.connect("database\\stock.db",timeout=timeout)
    else:
        raise ValueError("{} not supported".format(db_type))


def close_db(conn):
    conn.commit()
    conn.close()


def parse_config(path):
    with open(path) as f:
        split_symbol = "----"
        config_str = "".join(f.readlines())
        config_str = config_str.replace("\n","")
        config_str = config_str.replace("\t","  ")
        config_str = config_str.replace(split_symbol*2,split_symbol)
        config_str = config_str.strip(split_symbol)
        configs = dict([config.split("::") for config in config_str.split(split_symbol)])
    return configs


def parse_config1(path):
    with open(path) as f:
        split_symbol = "-"*4
        config_str = "".join(f.readlines()).strip(split_symbol)
        configs = []
        for config_table in config_str.split(split_symbol):
            configs.append(_parse_config_table(config_table))
    return dict(configs)


def _parse_config_table(config_table:str):
    split_symbol = ":"*2
    table_name, config_tab_details  = config_table.split(split_symbol)
    table_name = table_name.replace("\n","").replace(" ","")
    return table_name, _parse_config_tab_details(config_tab_details)


def _parse_config_tab_details(config_tab_details:str):
    split_symbol = "-"*2
    config_cols, config_others = config_tab_details.split(split_symbol)
    pass


def init_table(table_name, db_type):
    conn = connect_db(db_type)
    cursor = conn.cursor()

    configs = parse_config(path="database\\config\\{}".format(table_name))
    sql_drop = "drop table {}".format(table_name)
    print(sql_drop)
    sql_create = configs["create"].format(table_name)
    print(sql_create)
    try:
        cursor.execute(sql_drop)
    except Exception as e:
        print(e)
        pass
    cursor.execute(sql_create)
    close_db(conn)


def cols_from_cur(cursor):
    return tuple(desc[0] for desc in cursor.description)


def _sql_insert(db_type: str, table_name: str, cols: [str]):
    placeholders = {"mysql": "%s", "sqlite3": "?"}
    if db_type in placeholders:
        sql_insert = "INSERT INTO {} ".format(table_name) + "({})".format(
            ",".join(cols)) + "VALUES ({})"
        return sql_insert.format(",".join([placeholders[db_type]] * len(cols)))
    else:
        raise ValueError("{} not supported".format(db_type))


def _sql_replace(db_type: str, table_name: str, cols: [str]):
    # placeholders = {"mysql": "%s", "sqlite3": "?"}
    DB_INFOS = {"mysql":
                    {"placeholder":"%s","replace_clause":"REPLACE INTO"},
                "sqlite3":
                    {"placeholder":"?","replace_clause":"INSERT OR REPLACE "
                                                        "INTO"}}
    if db_type in DB_INFOS:
        sql_insert = "{0} {1} "\
                         .format(DB_INFOS[db_type]["replace_clause"],table_name) \
                     + "({})".format(", ".join(cols)) \
                     + " VALUES ({})"\
                         .format(", ".join(DB_INFOS[db_type]["placeholder"]*len(cols)))

        return sql_insert
    else:
        raise ValueError("{} not supported".format(db_type))


def get_latest_date(table,code, db_type:str):
    conn = connect_db(db_type=db_type)
    cursor = conn.cursor()
    cursor.execute(
        "select date from {0} where code='{1}'".format(table, code))
    rs = cursor.fetchall()
    close_db(conn)
    if len(rs) > 0:
        return sorted(rs, reverse=True)[0][0]
    else:
        return None


def get_trading_dates(db_type="sqlite3", cursor=None):
    # Get all trading dates from stock index table.
    if not cursor:
        conn = connect_db(db_type)
        cursor = conn.cursor()

    cursor.execute("select * from {0}".format(INDEX_DAY[TABLE]))
    df_idx_day = pd.DataFrame(cursor.fetchall())
    df_idx_day.columns = cols_from_cur(cursor)
    dates = sorted(df_idx_day["date"].unique())
    return dates


def get_all_stocks(db_type="sqlite3", cursor=None):
    # Get all stock codes from stock table.
    if not cursor:
        conn = connect_db(db_type)
        cursor = conn.cursor()

    cursor.execute("select distinct code from {0}".format(STOCK_DAY[TABLE]))
    return [row[0] for row in cursor.fetchall()]


def get_all_indexes(db_type="sqlite3", cursor=None):
    # Get all index codes from index table.
    if not cursor:
        conn = connect_db(db_type)
        cursor = conn.cursor()

    cursor.execute("select distinct code from {0}".format(INDEX_DAY[TABLE]))
    return [row[0] for row in cursor.fetchall()]


def write2db(df:pd.DataFrame, table, cols, db_type="sqlite3",
             conn=None, close=True):
    if not conn:
        conn = connect_db(db_type)
    cursor = conn.cursor()

    write_failure = 0
    params = [tuple(row) for _,row in df[list(cols)].iterrows()]
    # print(params)
    try:
        # Row is a series and only accepts indexes of type list to get values.
        # It fails if given tuple indexes, that's why list(cols) is used.
        # tuple(row[list(cols)]) is used to prevent type error in method cursor.execute(sql, paras)
        # cursor.execute(
        #     _sql_insert(db_type, table_name=table, cols=cols),tuple(row[list(cols)]))
        # print(_sql_replace(db_type, table_name=table, cols=cols))
        cursor.executemany(_sql_replace(db_type, table_name=table, cols=cols),
                           params)
    except Exception as err:
        # Failure should not happen, because the original row is deleted.
        # However, if it does, print and count it.
        write_failure += 1
        print("write err",err)

    # for _, row in df.iterrows():
    #     if "date" in row.index:
    #         args = row["code"],row["date"]
    #     else:
    #         args = (row["code"],)
    #     # Try to delete the row first if exists.
    #     try:
    #         cursor.execute(("delete from {0} where code='{1}' "
    #                         "and date='{2}'").format(table,*args))
    #     except Exception as e:
    #         pass
    #
    #     try:
    #         # Row is a series and only accepts indexes of type list to get values.
    #         # It fails if given tuple indexes, that's why list(cols) is used.
    #         # tuple(row[list(cols)]) is used to prevent type error in method cursor.execute(sql, paras)
    #         # cursor.execute(
    #         #     _sql_insert(db_type, table_name=table, cols=cols),tuple(row[list(cols)]))
    #         cursor.executemany(_sql_replace(db_type, table_name=table,
    #                                         cols=cols),df[cols].values)
    #     except Exception as err:
    #         # Failure should not happen, because the original row is deleted.
    #         # However, if it does, print and count it.
    #         write_failure += 1
    #         print(err)
    #         continue
    if close:
        close_db(conn)
    else:
        conn.commit()
    print("-"*10,"\nWrite failure:{0}\n".format(write_failure))
    return conn,write_failure


def create_df(cursor, table_name, start=None, where_clause=None):
    if start:
        start_cond = "date>={0}".format(start)
        where_clause = where_clause+" and "+start_cond if where_clause else start_cond

    if where_clause:
        sql_select = "select * from {0}".format(table_name)+" where "+where_clause
    else:
        sql_select = "select * from {0}".format(table_name)

    print(sql_select)
    cursor.execute(sql_select)
    df = pd.DataFrame(cursor.fetchall())
    df.columns = cols_from_cur(cursor)
    return df


def get_df(table,db_type="sqlite3"):
    conn = connect_db(db_type)
    cursor = conn.cursor()
    cursor.execute("select * from {0}".format(table))

    df = pd.DataFrame(cursor.fetchall(), columns=cols_from_cur(cursor))
    close_db(conn)
    del conn,cursor
    return df


def update_db(table, cols, db_type="sqlite3",
             conn=None, close=True):
    import time
    t0 = time.time()
    # df = get_df(table=table,db_type=db_type)
    conn = connect_db(db_type=db_type)
    cursor = conn.cursor()
    df = create_df(cursor,table_name=table)
    close_db(conn)

    print(df.shape, time.time()-t0)
    print(df.info(memory_usage="deep"))
    # time.sleep(3)

    init_table(table_name=table,db_type=db_type)

    if df["date"].dtype=="object":
        df["date"] = df["date"].apply(lambda s:s.replace("-","")).astype(int)
    print(time.time()-t0)

    params = (tuple(row) for _,row in df[list(cols)].iterrows())
    # print(params)
    write_failure = 0
    conn = connect_db(db_type)
    cursor = conn.cursor()
    try:
        # Row is a series and only accepts indexes of type list to get values.
        # It fails if given tuple indexes, that's why list(cols) is used.
        # tuple(row[list(cols)]) is used to prevent type error in method cursor.execute(sql, paras)
        cursor.executemany(_sql_replace(db_type, table_name=table, cols=cols),
                           params)
    except Exception as err:
        # Failure should not happen, because the original row is deleted.
        # However, if it does, print and count it.
        write_failure += 1
        print("write err",err)

    print(time.time()-t0)

    if close:
        close_db(conn)
    else:
        conn.commit()
    print("-"*10,"\nWrite failure:{0}\n".format(write_failure))
    return conn,write_failure


if __name__ == '__main__':
    import constants as const
    update_db(table=const.STOCK_DAILY_BASIC[const.TABLE],
              cols=const.STOCK_DAILY_BASIC[const.COLUMNS])