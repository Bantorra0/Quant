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


def parse_config(path):
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


if __name__ == '__main__':
    pass

