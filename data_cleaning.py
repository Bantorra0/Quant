import pandas as pd


def fillna_stock_day(df_single_stock:pd.DataFrame, dates:[str]):
    df_changed = pd.DataFrame(columns=df_single_stock.columns).set_index("date")

    code = df_single_stock["code"].iloc[0]

    df_single_stock = df_single_stock.set_index("date").sort_index()

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
        raise ValueError("Dates erro: {}".format(set(stock_dates)-set(dates)))
    
    target_dates = sorted(set(dates)-set(stock_dates))
    for d in target_dates:
        is_changed = False

        prev_d = dates[dates.index(d)-1]

        cols1, cols2 = ["open", "high", "low", "close"], ["vol", "amt"]
        cols = cols1 + cols2
        row = None
        if d in df_single_stock.index:
            row = df_single_stock.loc[d].copy()

            cnt_na = sum(df_single_stock.loc[d, cols].notnull())
            if 0 < cnt_na < len(cols):
                raise ValueError("Error row: {}".format(row))

            if df_single_stock.loc[d, cols].isnull().all().all():
                df_single_stock.loc[d, cols1] = df_single_stock.loc[
                    prev_d, cols1]
                df_single_stock.loc[d, cols2] = [0,0]
                is_changed = True

            if df_single_stock.loc[d, ["adj_factor"]].isnull().all(
            ).all():
                df_single_stock.loc[d, "adj_factor"] = \
                    df_single_stock.loc[prev_d, "adj_factor"]
                is_changed = True

        else:
            df_single_stock.loc[d] = df_single_stock.loc[prev_d]
            df_single_stock.loc[d, cols2] = [0, 0]
            is_changed = True

        if is_changed:
            df_changed.loc[d] = df_single_stock.loc[d]

    # start_idx = dates.index(start_date)
    # for i in range(start_idx + 1, len(dates)):
    #     is_changed = False
    # 
    #     # Skip if the row exists and the whole row is not null.
    #     # Put it at the beginning to reduce computation because it is the
    #     # most common case.
    #     if dates[i] in df_single_stock.index and df_single_stock.loc[
    #         dates[i]].notnull().all().all():
    #         continue
    # 
    #     cols1, cols2 = ["open", "high", "low", "close"], ["vol", "amt"]
    #     cols = cols1 + cols2
    #     row = None
    #     if dates[i] in df_single_stock.index:
    #         row = df_single_stock.loc[dates[i]].copy()
    # 
    #         cnt_na = sum(df_single_stock.loc[dates[i], cols].notnull())
    #         if 0 < cnt_na < len(cols):
    #             raise ValueError("Error row: {}".format(row))
    # 
    #         if df_single_stock.loc[dates[i], cols].isnull().all().all():
    #             df_single_stock.loc[dates[i], cols1] = df_single_stock.loc[
    #                 dates[i - 1], cols1]
    #             df_single_stock.loc[dates[i], cols2] = [0,0]
    #             is_changed = True
    # 
    #         if df_single_stock.loc[dates[i], ["adj_factor"]].isnull().all(
    #         ).all():
    #             df_single_stock.loc[dates[i], "adj_factor"] = \
    #                 df_single_stock.loc[dates[i-1], "adj_factor"]
    #             is_changed = True
    # 
    #     else:
    #         df_single_stock.loc[dates[i]] = df_single_stock.loc[dates[i - 1]]
    #         df_single_stock.loc[dates[i], cols2] = [0, 0]
    #         is_changed = True
    # 
    #     if is_changed:
    #         df_changed.loc[dates[i]] = df_single_stock.loc[dates[i]]
    #         # print("-" * 10 + "\n", dates[i], code)
    #         # print(dict(df_single_stock.loc[dates[i - 1]]))
    #         # print(None if row is None else dict(row))
    #         # print(dict(df_single_stock.loc[dates[i]]), "\n")

    return df_changed.reset_index()
