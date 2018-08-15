import db_operations as dbop
import df_operations as dfop
import collect as clct
import pandas as pd

def main():
    db_type = "sqlite3"
    #
    # init_table(STOCK_DAY.TABLE_NAME, db_type)
    # collect_stock_day(STOCK_DAY.pools(),db_type)
    #
    # init_table(INDEX_DAY.TABLE_NAME, db_type)
    # collect_index_day(INDEX_DAY.pools(), db_type)


    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    sql_select_all = "select * from {}"

    stock_day, index_day = clct.STOCK_DAY[clct.TABLE], clct.INDEX_DAY[clct.TABLE]

    cursor.execute(sql_select_all.format(stock_day))
    df_stck_d = pd.DataFrame(cursor.fetchall())
    df_stck_d.columns = dbop.cols_from_cur(cursor)

    cursor.execute(sql_select_all.format(index_day))
    df_idx_d = pd.DataFrame(cursor.fetchall())
    df_idx_d.columns = dbop.cols_from_cur(cursor)


    df_stck_d = df_stck_d.set_index("date")
    df_idx_d = df_idx_d.set_index("date")

    df = df_stck_d

    days = 20
    df["max_20"]=df["high"].rolling(window=days).max()
    df["min_20"]=df["low"].rolling(window=days).min()

    cols = ["open","high","low","close"]

    df_p_1 = df[cols].iloc[1:].copy()
    df_p_1.index = df.index[:-1]

    df_p_list = [("p1",df_p_1)]

    for name,group in df_p_list:
        cols = group.columns
        group.columns = list(map(lambda col:str(name)+"_"+col, cols))
        df = df.join(group,how="left")
        print(df.shape)
        print(df.iloc[:5])


    for name,group in df_idx_d.groupby("code"):
        cols = group.columns
        group.columns = list(map(lambda col:str(name)+"_"+col, cols))
        df = df.join(group,how="left")
        print(df.shape)
        print(df.iloc[:5])

    print(df.columns)




if __name__ == '__main__':
    main()