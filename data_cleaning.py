import pandas as pd
import db_operations as dbop
from constants import STOCK_DAY, TABLE, COLUMNS
import constants as const


def fillna_single_stock_day(df_single_stock:pd.DataFrame, dates:[str]):
    dates = sorted(dates)
    # df_single_stock = df_single_stock.sort_values("date")

    df_changed = pd.DataFrame(columns=df_single_stock.columns).set_index("date")
    code = df_single_stock["code"].iloc[0]

    # start_date is the first date on which the whole row is not null in the
    #  table.
    # stock_dates are the dates on which the whole row is not null in the table.
    stock_dates = df_single_stock.index[df_single_stock.notnull().all(axis=1)]
    start_date = None
    na_dates = []
    for d in stock_dates:
        if d not in dates:
            na_dates.append(d)
        else:
            start_date = d
            break
    if not start_date:
        print(stock_dates)
        print(df_single_stock)
        raise ValueError("start_date does not exist!")
    print(code, len(na_dates), na_dates[0] if na_dates else None,
          na_dates[-1] if na_dates else None)
    print("start:", start_date, "end:", dates[-1])
    
    stock_dates = stock_dates[stock_dates>=start_date]
    dates = dates[dates.index(start_date):]
    if len(set(stock_dates)-set(dates))>0:
        raise ValueError("Dates error: {} are in df_single_stock, but not in dates".format(set(stock_dates)-set(dates)))
    
    target_dates = sorted(set(dates)-set(stock_dates))
    for d in target_dates:
        is_changed = False

        prev_d = dates[dates.index(d)-1]

        cols1, cols2 = ["open", "high", "low", "close"], ["vol", "amt"]
        cols = cols1 + cols2
        row = None
        if d in df_single_stock.index:
            row = df_single_stock.loc[d].copy()

            cnt_null = sum(df_single_stock.loc[d, cols].isnull())
            if 0 < cnt_null < len(cols):
                raise ValueError("Error row: {}".format(row))

            if df_single_stock.loc[d, cols].isnull().all().all():
                df_single_stock.loc[d, cols1] = list(df_single_stock.loc[
                    prev_d, ["close"]])*4
                df_single_stock.loc[d, cols2] = [0,0]
                is_changed = True

            if df_single_stock.loc[d, ["adj_factor"]].isnull().all(
            ).all():
                df_single_stock.loc[d, "adj_factor"] = \
                    df_single_stock.loc[prev_d, "adj_factor"]
                is_changed = True

        else:
            new_row = [code]+list(df_single_stock.loc[prev_d, ["close"]])*4+[0,0,df_single_stock.loc[prev_d,"adj_factor"]]
            df_single_stock.loc[d] = new_row
            is_changed = True

        if is_changed:
            df_changed.loc[d] = df_single_stock.loc[d]

    return df_changed.reset_index()


def fillna_stock_day(df_stock_day:pd.DataFrame=None,dates = None,start="2000-01-01",db_type="sqlite3", conn=None):
    # Connect database if no connect object is passed.
    if dates is None or df_stock_day is None:
        if not conn:
            conn = dbop.connect_db(db_type)
        cursor = conn.cursor()

        if df_stock_day is None:
            # Read table stock_day.
            df_stock_day = dbop.create_df(cursor,const.STOCK_DAY[const.TABLE], start=start)
            print("\n"+"-"*10+"Data cleaning"+"-"*10)
            print(df_stock_day.shape)

        # Get all trading dates.
        if dates is None:
            dates = sorted(dbop.get_trading_dates(cursor=cursor))

    dates = sorted(dates)
    if start:
        df_stock_day = df_stock_day[df_stock_day.index>=start]
        dates = dates[dates.index(start):]

    # Check and fill null in table stock_day.
    print("start:", dates[0],", end:",dates[-1])
    print("{} trading days in total".format(len(dates)))
    change_list = []
    for code, df_single_stock in df_stock_day.groupby(by="code"):
        df_changed = fillna_single_stock_day(df_single_stock,dates)
        if len(df_changed)==0:
            print(code,": no change\n")
        else:
            change_list.append(df_changed)
            print(df_changed.shape,"\n")
            dbop.write2db(df_changed,STOCK_DAY[TABLE],STOCK_DAY[COLUMNS],
                          conn=conn, close=False)
    return change_list
