import sqlite3

import pymysql


def connect_db(db:str):
    if db=="mysql":
        return pymysql.connect(host='127.0.0.1', user='root', passwd='Bantorra',
                             db='quant', charset='utf8')
    elif db == "sqlite3":
        return sqlite3.connect("database\\stock.db")
    else:
        raise ValueError("{} not supported".format(db))


def close_db(conn):
    conn.commit()
    conn.close()


def _parse_config(path):
    with open(path) as f:
        split_symbol = "----"
        config_str = "".join(f.readlines())
        config_str = config_str.replace("\n","")
        config_str = config_str.replace("\t","  ")
        config_str = config_str.replace(split_symbol*2,split_symbol)
        config_str = config_str.strip(split_symbol)
        configs = dict([config.split("::") for config in config_str.split(split_symbol)])
    return configs


