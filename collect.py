import tushare as ts
import datetime

from db_operations import connect_db, _parse_config, close_db
from df_operations import natural_outer_join


TOKEN = 'ca7a0727b75dce94ad988adf953673340308f01bacf1a101d23f15fc'
STOCK_DAY = "stock_day"
INDEX_DAY = "index_day"
STOCK_COLUMNS = ["code", "date", "open", "high", "low", "close", "vol", "amt", "pre_close", "change", "pct_change", "adj_factor"]
INDEX_COLUMNS = ["code", "date", "open", "high", "low", "close", "vol"]


def _sql_insert(db:str, table_name:str):
    placeholders = {"mysql":"%s","sqlite3":"?"}
    if db in placeholders:
        sql_insert = "INSERT INTO {} ".format(table_name) \
                     + "({})".format(",".join(STOCK_COLUMNS)) \
                     + "VALUES ({})"
        return sql_insert.format(",".join([placeholders[db]] * len(STOCK_COLUMNS)))
    else:
        raise ValueError("{} not supported".format(db))


def _init_api(token=TOKEN):
    # 设置tushare pro的token并获取连接
    ts.set_token(token)
    return ts.pro_api()


def init_stock_day():
    conn = connect_db("sqlite3")
    cursor = conn.cursor()

    configs=_parse_config(path="database\\config\\stock_day")
    sql_drop = "drop table {}".format(STOCK_DAY)
    sql_create  = configs["create"]
    try:
        cursor.execute(sql_drop)
    finally:
        cursor.execute(sql_create)
    close_db(conn)


def init_table(table_name):
    conn = connect_db("sqlite3")
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


def unify_col_names(columns:list):
    columns = list(columns)
    if "ts_code" in columns:
        idx = columns.index("ts_code")
        columns[idx] = "code"
    if "trade_date" in columns:
        idx = columns.index("trade_date")
        columns[idx] = "date"
    return columns


def stock_pools():
    pools = ["002099.SZ","000581.SZ","000338.SZ"]
    return pools


def index_pools():
    return ["sh","sz"]


def collect_stock_day():
    pro = _init_api(TOKEN)
    conn = connect_db("sqlite3")
    cursor = conn.cursor()
    pools = stock_pools()
    for i,code in enumerate(pools):
        try:
            daily = pro.daily(ts_code=code)
            adj_factor = pro.adj_factor(ts_code=code)
            df = natural_outer_join(daily, adj_factor)
            # 打印进度
            print('Seq: ' + str(i + 1) + ' of ' + str(len(pools)) + '   Code: ' + str(code))
            df.columns = unify_col_names(df.columns)
            print(df.shape)

        except Exception as err:
            print(err)
            print('No DATA Code: ' + str(i))
            continue

        for _,row in df.iterrows():
            try:
                row["date"] = (datetime.datetime.strptime(row["date"], "%Y%m%d")).strftime('%Y-%m-%d')
                cursor.execute(_sql_insert("sqlite3",STOCK_DAY), list(row[STOCK_COLUMNS]))
                conn.commit()
            except Exception as err:
                print(err)
                continue
    close_db(conn)


def collect_index_day():
    conn = connect_db("sqlite3")
    cursor = conn.cursor()
    pools = index_pools()
    for i,code in enumerate(pools):
        try:
            df = ts.get_k_data(code=code)
            # 打印进度
            print('Seq: ' + str(i + 1) + ' of ' + str(len(pools)) + '   Code: ' + str(code))
            df.columns = unify_col_names(df.columns)
            print(df.shape)

        except Exception as err:
            print(err)
            print('No DATA Code: ' + str(i))
            continue

        for _,row in df.iterrows():
            cursor.execute(_sql_insert("sqlite3", INDEX_DAY), list(row[INDEX_COLUMNS]))
            conn.commit()
            # try:
            #     cursor.execute(_sql_insert("sqlite3",INDEX_DAY), list(row[INDEX_COLUMNS]))
            #     conn.commit()
            # except Exception as err:
            #     print(err)
            #     continue
    close_db(conn)


def main():
    init_table(INDEX_DAY)
    # init_table(STOCK_DAY)

    # collect_stock_day()
    collect_index_day()

    conn = connect_db("sqlite3")
    cursor = conn.cursor()
    # print(cursor.execute("select * from stock_day").fetchmany(100))
    print(cursor.execute("select * from index_day").fetchmany(100))


if __name__ == '__main__':
    main()

