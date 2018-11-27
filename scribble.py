import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# figs, ax = plt.subplots()
# x = np.arange(100)
# y = np.random.rand(100)
# line = ax.plot(x,y,label="line")
# ax.legend()
# plt.xticks([0,40,60,80])
# plt.show()


# import tushare as ts
# df = ts.get_k_data(code="sh",start="2000-01-01")
# print(df.iloc[0],df.iloc[-1])

import db_operations as dbop
import constants as const
conn = dbop.connect_db("sqlite3")
cursor = conn.cursor()
sql0 = "select * from {0} where date='2015-04-03'".format(const.STOCK_DAY[
                                                             const.TABLE])
cursor.execute(sql0)
df0 = pd.DataFrame(cursor.fetchall())
df0.columns = dbop.cols_from_cur(cursor)

# print(df)
codes = list(df0[df0.notna().all(axis=1)]["code"].unique())
print(len(codes))


sql1 = "select * from {0} where date='2018-11-26'".format(const.STOCK_DAY[
                                                             const.TABLE])
cursor.execute(sql1)
df1 = pd.DataFrame(cursor.fetchall())
df1.columns = dbop.cols_from_cur(cursor)
codes1 = list(df1["code"].unique())

v_list0 =[]
v_list1=[]
list0,list1=[],[]
for c in codes:
    v_list0.append(df0[df0["code"]==c]["close"].iloc[0] * df0[df0["code"]==c]["adj_factor"].iloc[0])
    v_list1.append(df1[df1["code"]==c]["close"].iloc[0] * df1[df1["code"]==c]["adj_factor"].iloc[0])
    print(c,df0[df0["code"]==c]["close"].iloc[0],df0[df0["code"]==c]["adj_factor"].iloc[0])
    print(c, df1[df1["code"] == c]["close"].iloc[0], df1[df1["code"] == c]["adj_factor"].iloc[0])

print(len(v_list0),len(v_list1))
print(sum(v_list0),sum(v_list1))
print(sum(v_list1)/sum(v_list0)*1000000)

print(set(codes1)-set(codes))


