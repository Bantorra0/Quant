import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants as const
import db_operations as dbop
import data_prepare as dp
import ml_model
import customized_obj as cus_obj

import xgboost.sklearn as xgb
import lightgbm.sklearn as lgbm
import sklearn.preprocessing as preproc
import sklearn.metrics as metrics
import sklearn as sk

import datetime
import time
import multiprocessing as mp

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
if __name__ == '__main__':
    db_type = "sqlite3"

    # targets = [{"period": 20, "fun": "max", "col": "high"},
    #            {"period": 20, "fun": "min", "col": "low"},
    #            {"period": 5, "fun": "max", "col": "high"},
    #            {"period": 5, "fun": "min", "col": "low"},
    #            # {"period": 20, "fun": "mean", "col": ""}
    #            ]
    #
    # time_delta = datetime.timedelta(days=1)
    # test_start = "2018-01-01"
    # train_length = 3000
    # max_feature_length = 1000
    #
    # cursor = dbop.connect_db(db_type=db_type).cursor()
    # num_files = 2
    #
    # df_stock_basic = dbop.create_df(cursor, const.STOCK_BASIC[const.TABLE])
    # h_stock_pool = sorted(df_stock_basic[df_stock_basic["is_hs"] == "H"]["code"])
    # s_stock_pool = sorted(df_stock_basic[df_stock_basic["is_hs"] == "S"]["code"])
    # print(len(h_stock_pool),len(s_stock_pool))
    #
    # train_bound = datetime.datetime.strptime(test_start, const.DATE_FORMAT) - train_length * time_delta
    # train_bound = datetime.datetime.strftime(train_bound, const.DATE_FORMAT)
    #
    # lower_bound = datetime.datetime.strptime(train_bound, const.DATE_FORMAT) - max_feature_length * time_delta
    # lower_bound = datetime.datetime.strftime(lower_bound, const.DATE_FORMAT)
    # print(test_start,train_bound,lower_bound)
    #
    # t0 = time.time()
    # # num_p = mp.cpu_count()
    # # p_pool = mp.Pool(processes=mp.cpu_count())
    # df_feature, df_not_in_X, cols_category, enc = ml_model.gen_data(
    #     targets=targets,
    #     lower_bound=lower_bound,
    #     start=train_bound,
    #     stock_pool=h_stock_pool)
    #
    # print("df_all:", df_feature.shape)
    # trading_date_idxes = df_feature.index.unique().sort_values(ascending=True)
    #
    # # X = ml_model.gen_X(df_feature, df_not_in_X.columns)
    # X = df_feature
    #
    # paras = [("y_l_rise", {"pred_period": 20, "is_high": True, "is_clf": False,"threshold":0.2}, df_not_in_X),
    #          ("y_l_decline", {"pred_period": 20, "is_high": False, "is_clf": False, "threshold":0.2}, df_not_in_X),
    #          ("y_s_rise", {"pred_period": 5, "is_high": True, "is_clf": False,"threshold":0.1}, df_not_in_X),
    #          ("y_s_decline", {"pred_period": 5, "is_high": False, "is_clf": False,"threshold":0.1}, df_not_in_X),
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
    # # with open(r"datasets/hgt_X.csv","w") as f:
    # #     X.to_csv(f)
    # # with open(r"datasets/hgt_Y.csv","w") as f:
    # #     Y.to_csv(f)
    # # with open(r"datasets/hgt_other_info.csv", "w") as f:
    # #     df_not_in_X.to_csv(f)
    #
    # # X.to_hdf(r"datasets/hgt_X.hdf",key="X")
    #
    # X.to_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")
    # Y.to_parquet(r"datasets/hgt_Y.parquet",engine="fastparquet")
    # df_not_in_X.to_parquet(r"datasets/hgt_other_info.parquet",engine="fastparquet")
    #
    # print(time.time()-t0)
    # print(X.info(memory_usage='deep'))
    # print("float64:",list(X.columns[X.dtypes=="float64"]))
    # print("int64:",list(X.columns[X.dtypes == "int64"]))
    # print("object:",list(X.columns[X.dtypes == "object"]))
    #
    # X["code"] = df_not_in_X["code"]
    # X_latest_day = X.loc[trading_date_idxes[-1]]
    # print(sorted(X_latest_day.columns[X_latest_day.isnull().any(axis=0)]))
    # print(X_latest_day.shape)
    # for k,v in X_latest_day.isnull().sum().sort_index().iteritems():
    #     if v>0:
    #         print(k,v)
    # pd.set_option("display.max_columns",10)
    # print(X_latest_day[X_latest_day["(open/p40mv_10k_open-1)"].isnull()][[
    #     "code","open","close","(open/p60max_open-1)",
    #     "(open/p40mv_10k_open-1)"]])
    #
    # del X["code"]
    # print(X.info(memory_usage='deep'))
    # print(Y.dtypes)


    files ={
        "Y":r"datasets/hgt_Y.parquet",
        "X":r"datasets/hgt_X.parquet",
        "other":r"datasets/hgt_other_info.parquet"
    }

    t_start_read_Y = time.time()
    # X = pd.read_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")
    Y = pd.read_parquet(files["Y"], engine="fastparquet")
    print("Reading parquet file {0} in {1:.2f}".format(files["Y"],time.time() - t_start_read_Y))
    print(Y.shape, Y.columns)
    print(Y.iloc[:5])

    N = len(Y.index)
    row_i_idxes = np.arange(N)
    for i in range(4):
        np.random.shuffle(row_i_idxes)

    k_split = 100
    subsample_idxes = row_i_idxes[:(N // k_split)]
    Y_subsample = Y.iloc[subsample_idxes]
    del Y

    t_start_read_X = time.time()
    X = pd.read_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")
    print("Reading parquet file {0} in {1:.2f}".format(files["X"],time.time() - t_start_read_X))
    X_subsample = X.iloc[subsample_idxes]
    del X

    t_start_read_other = time.time()
    df_other = pd.read_parquet(r"datasets/hgt_other_info.parquet",engine="fastparquet")
    print("Reading parquet file {0} in {1:.2f}".format(files["other"],time.time() - t_start_read_other))
    df_other_subsample = df_other.iloc[subsample_idxes]
    del df_other

    test_start = "2018-08-01"
    trading_dates = Y_subsample.index.unique().sort_values(ascending=True)
    train_dates = trading_dates[trading_dates<test_start][:-21]
    test_dates = trading_dates[trading_dates>=test_start]

    X_train = X_subsample.loc[train_dates]
    Y_train = Y_subsample.loc[train_dates]
    X_test = X_subsample.loc[test_dates]
    Y_test = Y_subsample.loc[test_dates]

    ycol = "y_l"
    cond = Y_test[ycol].notnull()
    X_test = X_test[cond]
    Y_test = Y_test[cond]
    print(X_train.shape,X_test.shape)

    reg = lgbm.LGBMRegressor(n_estimators=50,num_leaves=31,max_depth=12,
                             min_child_samples=30,random_state=0)

    train_start = time.time()
    cols_category = ["area", "industry", "market", "exchange", "is_hs"]
    reg.fit(X_train,Y_train[ycol],categorical_feature=cols_category)
    print("Train time:", time.time() - train_start)
    print(reg.score(X_test,Y_test[ycol]))
    df_feature_importance = ml_model.get_feature_importance(reg,X_test.columns)
    print(df_feature_importance)

    ycol2 = "y_l_rise"
    ml_model.pred_interval_summary(reg, X_test, Y_test[ycol])
    ml_model.pred_interval_summary(reg, X_test, Y_test[ycol2])
    plt.show()

    # Y_test_pred_reg={ycol:reg.predict(X_test)}
    #
    # interval = 0.05
    # n = int(1 / interval)
    # x0 = np.arange(n + 1) * interval
    # y01 = np.ones(x0.shape) * Y_test[ycol].mean()
    #
    # ycol2 = "y_l_rise"
    # y02 = np.ones(x0.shape) * Y_test[ycol2].mean()
    #
    # y1 = []
    # y2 = []
    # cnt = []
    # for i in range(-n, n):
    #     p0 = i * interval
    #     p1 = (i + 1) * interval
    #     cond = (p0 < Y_test_pred_reg[ycol]) & (Y_test_pred_reg[ycol] <= p1)
    #     cnt.append(sum(cond))
    #     y1.append((Y_test[ycol][cond].mean(), Y_test[ycol][cond].median(),
    #                Y_test[ycol][cond].std(), Y_test[ycol][cond].max(),
    #                Y_test[ycol][cond].min()))
    #     y2.append((Y_test[ycol2][cond].mean(), Y_test[ycol2][cond].median(),
    #                Y_test[ycol2][cond].std(), Y_test[ycol2][cond].max(), Y_test[
    #                    ycol2][cond].min()))
    # for c, p in zip(cnt, y1):
    #     print(c, p)
    #
    # plt.figure()
    # plt.bar(np.arange(-n, n) * interval + interval / 2, [mean for mean, _, _, _, _ in y1],
    #         width=0.8 * interval)
    #
    # plt.plot(x0, y01, color='r')
    # # plt.plot(x,y1,color='r')
    # plt.xlim(-1, 1)
    # plt.ylim(-0.5, 0.5)
    #
    # print()
    # for c, p in zip(cnt, y2):
    #     print(c, p)
    # plt.figure()
    # plt.bar(np.arange(-n, n) * interval + interval / 2, [mean for mean, _, _, _, _ in y2],
    #         width=0.8 * interval)
    #
    # plt.plot(x0, y02, color='r')
    # # plt.plot(x,y1,color='r')
    # plt.xlim(-1, 1)
    # plt.ylim(-0.5, 0.5)
    #
    # plt.show()







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
# print(ts.get_k_data(code="sh", start="2018-12-25", end=None))


# -------------------
# import multiprocessing as mp
# import db_operations as dbop
# import pickle
#
# def gen_df(*args):
#     return pd.DataFrame(np.arange(20).reshape(5,4),columns=list("abcd"))
#
#
# if __name__ == '__main__':
#     mp.freeze_support()
#     conn = dbop.connect_db("sqlite3")
#     pool = mp.Pool(processes=1)
#     res = pool.apply_async(func=gen_df,args=(1,))
#     print(pickle.dumps(conn))
#     df = res.get(timeout=10)
#     print(df)




# ------------------------
# if __name__ == '__main__':
#     cursor = dbop.connect_db("sqlite3").cursor()
#     t0 = time.time()
#     df = dbop.create_df(cursor,const.STOCK_DAY[const.TABLE], start="2018-10-01",where_clause="code='000008.SZ'")
#     # df = dbop.create_df(cursor,const.STOCK_DAY[const.TABLE])
#     pd.set_option("display.max_columns",10)
#     print(df[df["code"]=="000008.SZ"][["code","date","open","high","low","close","vol","amt","adj_factor"]]
#           .set_index("date")
#           .sort_index(ascending=False)
#           .iloc[:60])
#     print(time.time()-t0)

