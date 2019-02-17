import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants as const
import db_operations as dbop
import data_prepare as dp
import io_operations
import ml_model
import customized_obj as cus_obj
import io_operations as IO_op

import xgboost.sklearn as xgb
import lightgbm as lgbm
import sklearn.preprocessing as preproc
import sklearn.metrics as metrics
import sklearn as sk

import datetime
import time
import multiprocessing as mp
import pickle

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
    # db_type = "sqlite3"
    #
    # targets = [{"period": 20, "fun": "max", "col": "high"},
    #            {"period": 20, "fun": "min", "col": "low"},
    #            {"period": 5, "fun": "max", "col": "high"},
    #            {"period": 5, "fun": "min", "col": "low"},
    #            # {"period": 20, "fun": "mean", "col": ""}
    #            ]
    #
    # # time_delta = datetime.timedelta(days=1)
    # # test_start = "2018-01-01"
    # # train_length = 3000
    # # max_feature_length = 1000
    #
    # cursor = dbop.connect_db(db_type=db_type).cursor()
    #
    # df_stock_basic = dbop.create_df(cursor, const.STOCK_BASIC[const.TABLE])
    # h_stock_pool = sorted(df_stock_basic[df_stock_basic["is_hs"] == "H"]["code"])
    # s_stock_pool = sorted(df_stock_basic[df_stock_basic["is_hs"] == "S"]["code"])
    # print(len(h_stock_pool),len(s_stock_pool))
    #
    # # train_bound = datetime.datetime.strptime(test_start, const.DATE_FORMAT) - train_length * time_delta
    # # train_bound = datetime.datetime.strftime(train_bound, const.DATE_FORMAT)
    # # lower_bound = datetime.datetime.strptime(train_bound, const.DATE_FORMAT) - max_feature_length * time_delta
    # # lower_bound = datetime.datetime.strftime(lower_bound, const.DATE_FORMAT)
    # # print(test_start,train_bound,lower_bound)
    #
    # history_length = 10
    # start_year = 2016
    # end_year = 2018
    # dates = ["01-01","07-01"]
    # # for i in range(start_year,end_year+1):
    # #     for i,date in enumerate(dates):
    # #         start = str(start_year)+"-"+date
    # #         if i==len(dates)-1:
    # #             end = str(start_year+1)+"-"+dates[0]
    # #         else:
    # #             end = str(start_year)+"-"+dates[i+1]
    # #         upper_bound =
    #
    #
    # lower_bound="2013-01-01"
    # start = "2018-01-01"
    # end = "2018-07-01"
    # upper_bound="2019-01-01"
    #
    # t0 = time.time()
    # # num_p = mp.cpu_count()
    # # p_pool = mp.Pool(processes=mp.cpu_count())
    # df_feature, df_not_in_X, cols_category, enc = ml_model.gen_data(
    #     targets=targets,
    #     lowerbound=lower_bound,
    #     start=start,
    #     end=end,
    #     upperbound=upper_bound,
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
    # print(X.info(memory_usage='deep'))
    # print("float64:",list(X.columns[X.dtypes=="float64"]))
    # print("int64:",list(X.columns[X.dtypes == "int64"]))
    # print("object:",list(X.columns[X.dtypes == "object"]))
    #
    # f_path = r"datasets/stock_d.hdf"
    # f_path = io_operations._add_suffix_to_file_names(
    #     f_path, datetime.datetime.now().strftime("%Y-%m-%d"))
    # year = start[:4]
    # start_M_D = start[5:].replace("-", "")
    # end_M_D = end[5:].replace("-", "")
    # key = "{0}/{1}-{2}".format(year, start_M_D, end_M_D)
    # print(key)
    # print(f_path)
    #
    # X.to_hdf(f_path,key="X/"+key)
    # Y.to_hdf(f_path,key="Y/"+key)
    #
    # print(df_not_in_X.info(memory_usage="deep"))
    # pd.set_option("display.max_columns",10)
    # print(df_not_in_X.iloc[np.array(df_not_in_X.isnull().any(axis=1)),
    #                        np.array(df_not_in_X.isnull().any(axis=0))])
    # print(df_not_in_X[["code","qfq_open","qfq_close","qfq_vol",
    #                    "qfq_avg",
    #                    "f1mv_qfq_avg"]].iloc[np.array(df_not_in_X[["qfq_avg", "f1mv_qfq_avg"]].isnull().any(axis=1))])
    # df_not_in_X[["qfq_avg", "f1mv_qfq_avg"]] = df_not_in_X[["qfq_avg",
    #                                                         "f1mv_qfq_avg"]].fillna(-1)
    # df_not_in_X["delist_date"] = df_not_in_X["delist_date"].fillna("")
    # df_not_in_X.to_hdf(f_path,key="other/"+key)
    #
    # store = pd.HDFStore(f_path)
    # print(store.keys())
    # print(store["X/"+key].shape)
    # print(store["Y/" + key].shape)
    # print(store["other/" + key].shape)



    # files = {0:r"datasets/hgt.hdf"}
    # suffix = datetime.datetime.now().strftime('%Y-%m-%d')
    # f_hdf_name = ml_model.add_suffix_to_file_names(files, suffix)[0]
    #
    # t0=time.time()
    # X.to_hdf(f_hdf_name, key="X")
    # print("Write X in hdf:",time.time()-t0)
    #
    # t0 = time.time()
    # Y.to_hdf(f_hdf_name, key="Y")
    # print("Write Y in hdf:",time.time()-t0)
    #
    # t0 = time.time()
    # df_not_in_X.to_hdf(f_hdf_name, key="other")
    # print("Write other in hdf:",time.time()-t0)
    #
    #
    # files = {
    #     "Y": r"datasets/hgt_Y.parquet",
    #     "X": r"datasets/hgt_X.parquet",
    #     "other": r"datasets/hgt_other.parquet"
    # }
    #
    # files = ml_model.add_suffix_to_file_names(files, suffix)
    #
    # t0 = time.time()
    # X.to_parquet(files["X"], engine="fastparquet")
    # print("Write X in parquet with fastparquet",time.time()-t0)
    #
    # t0=time.time()
    # Y.to_parquet(files["Y"], engine="fastparquet")
    # print("Write Y in parquet with fastparquet", time.time() - t0)
    #
    # t0=time.time()
    # df_not_in_X.to_parquet(files["other"], engine="fastparquet")
    # print("Write other in parquet with fastparquet", time.time() - t0)
    #
    # dataset_info = {"feature_names": list(X.columns), "target_names": list(Y.columns),
    #                 "other_names": list(df_not_in_X.columns)}
    #
    # print(time.time()-t0)
    #
    # # X["code"] = df_not_in_X["code"]
    # # X_latest_day = X.loc[trading_date_idxes[-1]]
    # # print(sorted(X_latest_day.columns[X_latest_day.isnull().any(axis=0)]))
    # # print(X_latest_day.shape)
    # # for k,v in X_latest_day.isnull().sum().sort_index().iteritems():
    # #     if v>0:
    # #         print(k,v)
    # # pd.set_option("display.max_columns",10)
    # # print(X_latest_day[X_latest_day["(open/p40mv_10k_open-1)"].isnull()][[
    # #     "code","open","close","(open/p60max_open-1)",
    # #     "(open/p40mv_10k_open-1)"]])
    # # del X["code"]
    #
    # print(X.info(memory_usage='deep'))
    # print(Y.dtypes)
    #
    # N = len(Y.index)
    # row_i_idxes = np.arange(N)
    # for i in range(4):
    #     np.random.shuffle(row_i_idxes)
    #
    # dataset_info["shuffle"]= row_i_idxes
    # f_pickle_name = r"datasets/hgt_dataset_info"
    # f_pickle_name = ml_model.add_suffix_to_file_names({0:f_pickle_name},suffix)[0]
    # with open(f_pickle_name, mode="wb") as f:
    #     t0 = time.time()
    #     pickle.dump(dataset_info, f)
    #     print("Pickle dump time:",time.time()-t0)
    #
    # t0 = time.time()
    # X=pd.read_hdf(f_hdf_name, "X")
    # print("Read X in hdf5:",time.time()-t0)
    # print(X.shape)
    # t0 = time.time()
    # Y=pd.read_hdf(f_hdf_name, "Y")
    # print("Read Y in hdf5:",time.time()-t0)
    # print(Y.shape)
    # t0 = time.time()
    # df_other_info = pd.read_hdf(f_hdf_name, "other")
    # print("Read other info in hdf5:", time.time() - t0)
    # print(df_other_info.shape)
    #
    # t0 = time.time()
    # X = pd.read_parquet(files["X"])
    # print("Read X in parquet:", time.time() - t0)
    # print(X.shape)
    # t0 = time.time()
    # Y = pd.read_parquet(files["Y"])
    # print("Read Y in parquet:", time.time() - t0)
    # print(Y.shape)
    # t0 = time.time()
    # df_other_info = pd.read_parquet(files["other"])
    # print("Read other info in parquet:", time.time() - t0)
    # print(df_other_info.shape)
    #
    # t0 = time.time()
    # X = pd.read_parquet(files["X"],engine="fastparquet")
    # print("Read X in parquet,engine=fastparquet:", time.time() - t0)
    # print(X.shape,X.info(memory_usage="deep"))
    # t0 = time.time()
    # Y = pd.read_parquet(files["Y"],engine="fastparquet")
    # print("Read Y in parquet,engine=fastparquet:", time.time() - t0)
    # print(Y.shape)
    # t0 = time.time()
    # df_other_info = pd.read_parquet(files["other"],engine="fastparquet")
    # print("Read other info in parquet,engine=fastparquet:", time.time() - t0)
    # print(df_other_info.shape)




    # files ={
    #     "Y":r"datasets/hgt_Y.parquet",
    #     "X":r"datasets/hgt_X.parquet",
    #     "other":r"datasets/hgt_other.parquet"
    # }
    #
    # with open(r"datasets/hgt_dataset_info", mode="rb") as f:
    #     dataset_info = pickle.load(f)
    # row_i_idxes = dataset_info["shuffle"]
    # N = len(row_i_idxes)
    # k_split = 100
    # print(row_i_idxes)
    # subsample_idxes = row_i_idxes[:(N // k_split)]
    #
    # t_start_read_Y = time.time()
    # # X = pd.read_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")
    # Y = pd.read_parquet(files["Y"], engine="fastparquet")
    # print("Reading parquet file {0} in {1:.2f}".format(files["Y"],time.time() - t_start_read_Y))
    # print(Y.shape, Y.columns)
    # # print(Y.iloc[:5])
    # Y_subsample = Y.iloc[subsample_idxes]
    # del Y
    #
    # t_start_read_X = time.time()
    # X = pd.read_parquet(r"datasets/hgt_X.parquet",engine="fastparquet")
    # print("Reading parquet file {0} in {1:.2f}".format(files["X"],time.time() - t_start_read_X))
    # X_subsample = X.iloc[subsample_idxes]
    # del X
    #
    # t_start_read_other = time.time()
    # df_other = pd.read_parquet(r"datasets/hgt_other.parquet",engine="fastparquet")
    # print("Reading parquet file {0} in {1:.2f}".format(files["other"],time.time() - t_start_read_other))
    # df_other_subsample = df_other.iloc[subsample_idxes]
    # del df_other
    #

    X, Y, _ = IO_op.read_hdf5(start="2013-01-01",end="2019-01-01",
                                     subsample="5-0")
    print(X.info(memory_usage="deep"))
    del X["industry"]
    test_start = "2018-07-01"
    trading_dates = Y.index.unique().sort_values(ascending=True)
    train_dates = trading_dates[trading_dates<test_start][:-21]
    test_dates = trading_dates[trading_dates>=test_start][:-21]

    X_train = X.loc[train_dates]
    Y_train = Y.loc[train_dates]
    X_test = X.loc[test_dates]
    Y_test = Y.loc[test_dates]
    del X,Y

    ycol = "y_l"
    cond = Y_test[ycol].notnull()
    X_test = X_test[cond]
    Y_test = Y_test[cond]
    print(X_train.shape,X_test.shape)

    reg1 = lgbm.LGBMRegressor(n_estimators=50, num_leaves=32, max_depth=12,
                              min_child_samples=30, random_state=0)

    train_start = time.time()
    # cols_category = ["area", "industry", "market", "exchange", "is_hs"]
    cols_category = ["area", "market", "exchange", "is_hs"]
    reg1.fit(X_train, Y_train[ycol], categorical_feature=cols_category)
    print("Train time:", time.time() - train_start)
    print(reg1.score(X_test, Y_test[ycol]))
    df_feature_importance = ml_model.get_feature_importance(reg1, X_test.columns)
    pd.set_option("display.max_columns",10)
    pd.set_option("display.max_rows",256)
    print(df_feature_importance[df_feature_importance[
        "importance_raw"]>0].round({
        "importance_percent":2}).iloc[:10])

    ycol2 = "y_l_rise"
    ml_model.pred_interval_summary(reg1, X_test, Y_test[ycol])
    ml_model.pred_interval_summary(reg1, X_test, Y_test[ycol2])
    y_pred1 = reg1.predict(X_test)

    # Reg2, learn buy pct.
    y_test = Y_test[ycol]
    initial_learning_rate = 0.5
    decay_learning_rate = lambda n: initial_learning_rate / (1 + n / 25)
    f = lgbm.reset_parameter(learning_rate=decay_learning_rate)
    reg2 = lgbm.LGBMRegressor(n_estimators=400,num_leaves=32,max_depth=8,
                             min_child_samples=30,random_state=0,
                              learning_rate=initial_learning_rate,
                              objective=cus_obj.custom_revenue_obj)
    t0 = time.time()
    X_train["pred"] = reg1.predict(X_train)
    X_test["pred"] = reg1.predict(X_test)
    reg2.fit(X_train, Y_train[ycol], categorical_feature=cols_category,
             callbacks=[f]
             )
    print("Training time: {0}".format(time.time()-t0))

    y_pred2 = reg2.predict(X_test)
    sigmoid = pd.Series(1 / (1 + np.exp(-y_pred2)))
    plt.figure()
    plt.hist(sigmoid)
    idx = sigmoid.index[sigmoid > 0.8]
    result2 = pd.DataFrame()
    result2["buy_pct"] = sigmoid
    result2["min_max"] = Y_test[ycol].values
    result2["increase"] = Y_test[ycol2].values
    r, revenue, tot_revenue = cus_obj.get_revenue(Y_test[ycol], y_pred2)
    result2["return_rate"] = r.values
    result2["revenue"] = revenue.values
    print(result2[result2["buy_pct"] > 0.8])

    print(reg2)
    print(tot_revenue, sum(r * 0.5))
    y_train_pred2 = reg2.predict(X_train)
    r_train,revenue_train,tot_revenue_train = cus_obj.get_revenue(Y_train[
                                                                      ycol],y_train_pred2)
    print("train:", tot_revenue_train, sum(r_train*0.5))
    for i in range(1, 10):
        threshold = i * 0.1
        print(">{0}:".format(threshold),
              result2[result2["buy_pct"] > threshold]["revenue"].sum(),
              sum(result2["buy_pct"] > threshold))

    # Show reg1, learn y_l.
    result1 = pd.DataFrame()
    result1["pred"] = y_pred1
    result1["increase"] = Y_test[ycol].values
    result1["return_rate"] = r.values
    for i in range(0, 11):
        threshold = i * 0.05
        print("\n>{0}:".format(threshold))
        # print(result1[result1["pred"] > threshold])
        print(result1[result1["pred"] > threshold]["return_rate"].sum(),sum(result1["pred"] > threshold))
        cond1 = (result1["pred"] > threshold) & (result2["buy_pct"]>0.8)
        print(result1[cond1]["return_rate"].sum(),sum(cond1))

        cond2 = (result1["pred"] > threshold) & (result2["buy_pct"] > 0.9)
        print(result1[cond2]["return_rate"].sum(), sum(cond2))

    plt.show()








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

