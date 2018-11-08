import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x = -np.arange(100)
# y = np.random.rand(100)
# line = plt.plot(x,y)
# plt.xticks([0,40,60,80])
# plt.legend([line],["line"])
# plt.show()


# import tushare as ts
# df = ts.get_k_data(code="sh",start="2000-01-01")
# print(df.iloc[0],df.iloc[-1])

import db_operations as dbop
from constants import TOKEN, STOCK_DAY, INDEX_DAY,TABLE,COLUMNS
import pandas as pd
import data_cleaning as dc
import collect

db_type = "sqlite3"
conn = dbop.connect_db(db_type)
cursor = conn.cursor()

# Get all trading dates from stock index table.
cursor.execute("select * from {0}".format(
                    INDEX_DAY[TABLE]))
df_idx_day = pd.DataFrame(cursor.fetchall())
df_idx_day.columns = dbop.cols_from_cur(cursor)

# Get day data of all stocks from stocks table.
cursor.execute("select * from {0}".format(STOCK_DAY[TABLE]))
df_stocks_day = pd.DataFrame(cursor.fetchall())
df_stocks_day.columns = dbop.cols_from_cur(cursor)
print(df_stocks_day.shape, df_idx_day.shape)


df_stck_na = df_stocks_day[df_stocks_day.isna().any(axis=1)]
df_idx_na = df_idx_day[df_idx_day.isna().any(axis=1)]

print("indexes:\t",df_idx_na)
print("\nstocks:\t",df_stck_na.shape)



dates = sorted(df_idx_day["date"].unique())


# print(len(dates), dates[0],dates[-1])
# for code, df_single_stock in df_stocks_day.groupby(by="code"):
#     df_single_stock = df_single_stock.sort_index()
#     df_changed = dc.fillna_stock_day(df_single_stock,dates)
#     if len(df_changed)==0:
#         print(code,": no change\n")
#     else:
#         print(df_changed.shape,"\n")
#         dbop.write2db(df_changed,STOCK_DAY[TABLE],STOCK_DAY[COLUMNS])

for r in cursor.execute("select * from {0} where code = '002217.SZ' and "
                        "date>='{"
                        "1}'".format(STOCK_DAY[
                                                                    TABLE],
                                                                "2018-10-15")).fetchall():
    print(r)


# for df in collect.download_index_day(collect.idx_pools(), db_type,
#                                     update=True):
#     print(df)
# for df in collect.download_stock_day(collect.stck_pools(), db_type,
#                                      update=True):
#     print(df)
