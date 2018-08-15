import tushare as ts
from db_operations import connect_db, _parse_config, close_db
from df_operations import natural_outer_join
import datetime

TOKEN = 'ca7a0727b75dce94ad988adf953673340308f01bacf1a101d23f15fc'

TABLE,COLUMNS="table","columns"
STOCK_DAY = {TABLE:"stock_day",
             COLUMNS:("code","date","open","high","low","close","vol","amt",
                        "pre_close","p_change","pct_change","adj_factor")}
INDEX_DAY = {TABLE:"index_day",
             COLUMNS:("code","date","open","high","low","close","vol")}


def _sql_insert(db:str, table_name:str, cols:[str]):
    placeholders = {"mysql":"%s","sqlite3":"?"}
    if db in placeholders:
        sql_insert = "INSERT INTO {} ".format(table_name) \
                     + "({})".format(",".join(cols)) \
                     + "VALUES ({})"
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


def unify_col_names(columns:list):
    columns = list(columns)
    mapping = {"ts_code":"code", "trade_date":"date","amount":"amt",
               "volume":"vol", "change":"p_change"}
    for col1,col2 in mapping.items():
        if col1 in columns:
            idx = columns.index(col1)
            columns[idx] = col2
    return columns


def insert_to_db(row, db_type:str, table_name, columns):
    conn = connect_db(db_type)
    cursor = conn.cursor()
    cursor.execute(_sql_insert(db_type, table_name,columns),
                   list(row[columns]))
    conn.commit()


def collect_index_day(pools:[str], db_type:str):
    conn = connect_db(db_type)
    cursor = conn.cursor()
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
            try:
                cursor.execute(_sql_insert(db_type, INDEX_DAY.TABLE,
                                           INDEX_DAY.COLUMNS),
                               list(row[INDEX_DAY.COLUMNS]))
                conn.commit()
            except Exception as err:
                print(err)
                continue
    close_db(conn)


def collect_stock_day(pools:[str], db_type:str):
    pro = _init_api(TOKEN)
    conn = connect_db(db_type)
    cursor = conn.cursor()
    for i,code in enumerate(pools):
        try:
            daily = pro.daily(ts_code=code)
            adj_factor = pro.adj_factor(ts_code=code)
            df =  natural_outer_join(daily, adj_factor)
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
                cursor.execute(_sql_insert(db_type,STOCK_DAY.TABLE,
                                           STOCK_DAY.COLUMNS),
                               tuple(row[STOCK_DAY.COLUMNS]))
                conn.commit()

            except Exception as err:
                print(err)
                continue
    close_db(conn)


def main():
    db_type = "sqlite3"
    #
    # init_table(STOCK_DAY.TABLE_NAME, db_type)
    # collect_stock_day(STOCK_DAY.pools(),db_type)
    #
    # init_table(INDEX_DAY.TABLE_NAME, db_type)
    # collect_index_day(INDEX_DAY.pools(), db_type)

    conn = connect_db(db_type)
    cursor = conn.cursor()
    # print(cursor.execute("select * from stock_day").fetchmany(50))
    print(cursor.fetchmany(100))

    cursor.execute("select * from index_day")
    print(cursor.fetchmany(100))
    # print(cursor.execute("select * from stock_day").fetchmany(100))


if __name__ == '__main__':
    main()

