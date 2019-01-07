import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
# # figs, ax = plt.subplots()
# # x = np.arange(100)
# # y = np.random.rand(100)
# # line = ax.plot(x,y,label="line")
# # ax.legend()
# # plt.xticks([0,40,60,80])
# # plt.show()
#
#
# # import tushare as ts
# # df = ts.get_k_data(code="sh",start="2000-01-01")
# # print(df.iloc[0],df.iloc[-1])
#
# import db_operations as dbop
# import constants as const
# conn = dbop.connect_db("sqlite3")
# cursor = conn.cursor()
# sql0 = "select * from {0} where date='2015-06-05'".format(const.STOCK_DAY[
#                                                              const.TABLE])
# cursor.execute(sql0)
# df0 = pd.DataFrame(cursor.fetchall())
# df0.columns = dbop.cols_from_cur(cursor)
#
# # print(df)
# # codes = list(df0[df0.notna().all(axis=1)]["code"].unique())
# codes = ['600401.SH', '300292.SZ', '002335.SZ', '002446.SZ', '002099.SZ', '300059.SZ', '000539.SZ', '600023.SH', '600536.SH', '300038.SZ', '002402.SZ', '000070.SZ', '002217.SZ', '600845.SH', '600549.SH', '600966.SH', '300383.SZ', '000581.SZ', '000725.SZ', '300113.SZ', '600567.SH', '600050.SH', '300068.SZ', '002068.SZ', '600522.SH', '002463.SZ', '601006.SH', '601336.SH', '600345.SH', '002410.SZ', '000636.SZ', '600392.SH', '600703.SH', '002236.SZ', '000063.SZ', '000001.SZ', '600305.SH', '000488.SZ', '000338.SZ', '603799.SH', '600887.SH']
#
# print(len(codes))
#
#
# sql1 = "select * from {0} where date='2018-11-26'".format(const.STOCK_DAY[
#                                                              const.TABLE])
# cursor.execute(sql1)
# df1 = pd.DataFrame(cursor.fetchall())
# df1.columns = dbop.cols_from_cur(cursor)
# codes1 = list(df1["code"].unique())
#
# v_list0 =[]
# v_list1=[]
# list0,list1=[],[]
# df_change = pd.DataFrame(columns=["code","value"]).set_index("code")
# for c in codes:
#     v0 = df0[df0["code"]==c]["close"].iloc[0] * df0[df0["code"]==c]["adj_factor"].iloc[0]
#     v1 = df1[df1["code"]==c]["close"].iloc[0] * df1[df1["code"]==c]["adj_factor"].iloc[0]
#     v_list0.append(v0)
#     v_list1.append(v1)
#     print(c,df0[df0["code"]==c]["close"].iloc[0],df0[df0["code"]==c]["adj_factor"].iloc[0])
#     print(c, df1[df1["code"] == c]["close"].iloc[0], df1[df1["code"] == c]["adj_factor"].iloc[0])
#     df_change.loc[c] = v1/v0
#
# df_change = df_change.sort_values("value")
#
# print(len(v_list0),len(v_list1))
# print(sum(v_list0),sum(v_list1))
# print(sum(v_list1)/sum(v_list0)*1000000)
#
# print(set(codes1)-set(codes))
# print(list(df_change[df_change["value"]<1.3].index.unique()))
#
#


# def loss(y_true,y_pred):
#     x = -100 * y_true*(y_pred-y_true)
#     return 1/(1+np.exp(-x))
#
#
# if __name__ == '__main__':
#     y_true = np.arange(-1,4)*0.1
#     y_pred = np.array([-0.2,-0.05,-0.1,0.15,0.4])
#
#     print(loss(y_true,y_pred))


base_price = 1
change_rates = np.random.normal(0,0.03,size=10)
print(change_rates)
shape=11
open = np.ones(shape=11)*base_price
for i,chg in enumerate(change_rates):
    if chg<-0.1:
        chg = -0.1
    elif chg>0.1:
        chg=0.1
    open[i+1] = open[i] * (1+chg)

print(open)

high = open * (1+np.random.uniform(0,0.05,size=open.shape))
low = open * (1-np.random.uniform(0,0.05,size=open.shape))
close = open * (1+np.random.uniform(-0.025,0.025,size=open.shape))
print(high)
print(low)
print(close)