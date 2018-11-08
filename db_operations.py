import sqlite3
import pandas as pd
import pymysql


def connect_db(db_type:str):
    if db_type== "mysql":
        return pymysql.connect(host='127.0.0.1', user='root', passwd='Bantorra',
                             db='quant', charset='utf8')
    elif db_type == "sqlite3":
        return sqlite3.connect("database\\stock.db")
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


def write2db(df:pd.DataFrame, table, cols, db_type="sqlite3"):
    conn = connect_db(db_type)
    cursor = conn.cursor()
    write_failure = 0
    for _, row in df.iterrows():
        try:
            # Row is a series and only accepts indexes of type list to get values.
            # It fails if given tuple indexes, that's why list(cols) is used.
            # tuple(row[list(cols)]) is used to prevent type error in method cursor.execute(sql, paras)
            cursor.execute(
                _sql_insert(db_type, table_name=table, cols=cols),tuple(row[list(cols)]))
            conn.commit()
        except Exception as err:
            write_failure += 1
            print(err)
            continue


if __name__ == '__main__':
    cursor = connect_db("sqlite3").cursor()
    cursor.execute("select * from stock_day where date>='2018-08-15'")
    for row in cursor.fetchall():
        print(row)


