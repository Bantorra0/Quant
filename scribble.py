import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



# import collect
# import constants
#
# pro = collect._init_api(constants.TOKEN)
# status_list = ["L","D","P"] # L上市，D退市，P暂停上市。
# fields = ['ts_code',
#           'symbol',
#           'name',
#           'area',
#           'industry',
#           'fullname',
#           'enname',
#           'market',
#           'exchange',
#           'curr_type',
#           'list_status',
#           'list_date',
#           'delist_date',
#           'is_hs']
# df_list = []
# for status in status_list:
#     df_list.append(pro.stock_basic(list_status=status,fields=",".join(fields)))
#     print(df_list[-1].shape)
#
# df = pd.concat(df_list,sort=False,ignore_index=True)
# df.columns = collect.unify_col_names(df.columns)
# print(df)
# print(df.columns)
#
# df.to_excel("stock_basic.xlsx",index=False)


# base_price = 1
# change_rates = np.random.normal(0,0.03,size=10)
# print(change_rates)
# shape=11
# open = np.ones(shape=11)*base_price
# for i,chg in enumerate(change_rates):
#     if chg<-0.1:
#         chg = -0.1
#     elif chg>0.1:
#         chg=0.1
#     open[i+1] = open[i] * (1+chg)
#
# print(open)
#
# high = open * (1+np.random.uniform(0,0.05,size=open.shape))
# low = open * (1-np.random.uniform(0,0.05,size=open.shape))
# close = open * (1+np.random.uniform(-0.025,0.025,size=open.shape))
# print(high)
# print(low)
# print(close)





# --------------------------------

# import constants as const
# import db_operations as dbop
# import data_prepare as dp
# import ml_model
# import customized_obj as cus_obj
#
# import xgboost.sklearn as xgb
# import lightgbm.sklearn as lgbm
# import sklearn.preprocessing as preproc
# import sklearn.metrics as metrics
#
# import datetime
# import time
#
# targets = [{"period": 20, "fun": "max", "col": "high"},
#            {"period": 20, "fun": "min", "col": "low"},
#            # {"period": 5, "fun": "max", "col": "high"},
#            # {"period": 5, "fun": "min", "col": "low"},
#            # {"period": 20, "fun": "mean", "col": ""}
#            ]
#
# time_delta = datetime.timedelta(days=1)
# test_start = "2018-09-01"
# train_length = 1000
# max_feature_length = 250
#
# train_bound = datetime.datetime.strptime(test_start, const.DATE_FORMAT) - train_length * time_delta
# train_bound = datetime.datetime.strftime(train_bound, const.DATE_FORMAT)
#
# lower_bound = datetime.datetime.strptime(train_bound, const.DATE_FORMAT) - max_feature_length * time_delta
# lower_bound = datetime.datetime.strftime(lower_bound, const.DATE_FORMAT)
# print(test_start,train_bound,lower_bound)
#
# t0 = time.time()
# df_all, cols_future, cols_category,cols_not_for_model,enc = ml_model.gen_data(
#     targets=targets,
#                                         lower_bound=lower_bound,
#                                         start=train_bound,
#                                         stock_pool=None)
#
# print("df_all:", df_all.shape)
# trading_date_idxes = df_all.index.unique().sort_values(ascending=True)
#
# X = ml_model.gen_X(df_all, cols_future+cols_not_for_model)
#
# paras = [("y_l_rise", {"pred_period": 20, "is_high": True, "is_clf": False,"threshold":0.2}, df_all),
#          ("y_l_decline", {"pred_period": 20, "is_high": False, "is_clf": False, "threshold":0.2}, df_all),
#          # ("y_s_rise", {"pred_period": 5, "is_high": True, "is_clf": False,"threshold":0.1}, df_all),
#          # ("y_s_decline", {"pred_period": 5, "is_high": False, "is_clf": False,"threshold":0.1}, df_all),
#          ]
#
# # paras = [("y_l", {"pred_period": 20, "is_high": True, "is_clf": False,
# #                        "threshold":0.1}, df_all)]
# Y = pd.concat([ml_model.gen_y(v2, **v1) for k, v1, v2 in paras], axis=1)
# Y.columns = [k for k, _, _ in paras]
# Y.index = X.index
# Y["y_l"] = Y.apply(
#     lambda r:r["y_l_rise"] if r["y_l_rise"]> -r["y_l_decline"] else r["y_l_decline"],
#     axis=1)
# print(X.shape, Y.shape, Y.columns)
#
# print(time.time()-t0)
# print(X.info(memory_usage='deep'))


# ----------------------------
# import collect
# import constants as const
# import db_operations as dbop
# import multiprocessing as mp
# import threading
# db_type = "sqlite3"
#
# pro = collect._init_api(const.TOKEN)
# # df = pro.daily(ts_code='000651.SZ', start_date="20181225",end_date="")
# # kwargs = {'ts_code':'000651.SZ', 'start_date':'20181225','end_date':''}
#
# kwargs = {"code":'000651.SZ', "db_type":"sqlite3", "update":False,
#                              "start":"2000-01-01", "verbose":0,"conn":None}

# pool = mp.Pool(processes=2)
# res = pool.apply_async(func=pro.daily,kwds=kwargs)
# df = res.get()
# print(df)



# t = threading.Thread(target=pro.daily,kwargs=kwargs)
# t.start()

# cursor = dbop.connect_db("sqlite3").cursor()
# df_stock_basic = dbop.create_df(cursor, const.STOCK_BASIC[const.TABLE])
# print(df_stock_basic[df_stock_basic["is_hs"]!="N"][["code","is_hs"]])


# print(dbop.get_latest_date(const.STOCK_DAY[const.TABLE],"000",db_type))

# import tushare as ts
# print(ts.get_k_data(code='sh', start="2018-12-25", end=None))





# ------------------
import multiprocessing as mp
import time


def f(s=0):
    print("start f")
    time.sleep(10)
    print("hello",s)

if __name__ == '__main__':
    pool = mp.Pool(processes=1)
    for i in range(3):
        print("\nbefore apply_async")
        rec = pool.apply_async(func=f,args=(1,))
        print("after apply_async")
        time.sleep(1)
        try:
            rec.get(timeout=2)
        except mp.TimeoutError as err:
            print("Timeout",type(err))
