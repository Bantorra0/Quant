from daily_job import *
from ml_model import pred_interval_summary


if __name__ == '__main__':
    # X, _, df_other = IO_op.read_hdf5(start="2019-01-01", end="2020-01-01",
    #                                  # subsample="1-0"
    #                                  )
    # print(X.info(memory_usage="deep"))
    # del X["industry"]
    # # predict_dates = sorted(X.index.unique())[-20:]
    # fq_cols = ["open", "high", "low", "close", "avg", "vol"]
    # print(list(X.columns))
    # cols = ["code"]+[col+"0" for col in fq_cols]+["amt"]+["f1mv_open"]
    # df_all = pd.concat([X, df_other], axis=1)[cols]
    # df_all = df_all.reset_index().set_index(["code","date"]).sort_index()
    # idx = pd.IndexSlice
    # codes = ['603713.SH',
    #          '000806.SZ',
    #          '600919.SH',
    #          '603228.SH',
    #          '002879.SZ',
    #          '300134.SZ',
    #          '300045.SZ']
    # df = df_all.loc[idx[codes,:],:]
    # for code, group in df.groupby(level="code"):
    #     print(group.reset_index("code").loc[-20:])
    #
    # t0 = time.time()
    # r = get_return_rate(df.loc[idx[codes[0],:],fq_cols[:4]])
    # print("t1:",time.time()-t0)
    # t0 = time.time()
    # get_return_rate2(df.loc[idx[codes[0], :], fq_cols[:4]])
    # print("t2:", time.time() - t0)
    # print(r)

    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 256)

    cols_category = ["area", "market", "exchange", "is_hs"]
    ycol1, ycol2, ycol3, ycol4 = "y_l_r", "y_l", "y_l_avg", "y_l_rise"

    trading_dates = dbop.get_trading_dates()
    dates = [20160701, 20170101, 20170701, 20180101, 20180701, 20190101,
             20190701]
    start = 20130101
    # dates = [20180701,20190101]
    for test_start, test_end in zip(dates[:-1], dates[1:]):
        print("\ntest period:{0}-{1}".format(test_start, test_end))
        # test_start = 20180701

        # trading_dates = Y.index.unique().sort_values(ascending=True)
        # train_dates = trading_dates[trading_dates < test_start][:-21]
        # # train_start,train_end = min(train_dates),max(train_dates)
        # test_dates = trading_dates[
        #     (trading_dates >= test_start) & (trading_dates < test_end)]
        end = test_start

        reg_params = [{"n_estimators": 10, "learning_rate": 2, "num_leaves": 15,
                       "max_depth": 8, "objective": cus_obj.custom_revenue_obj,
                       "min_child_samples": 30, "random_state": 0, },
            {"n_estimators": 10, "learning_rate": 2, "num_leaves": 15,
             "max_depth": 8, "objective": cus_obj.custom_revenue2_obj,
             "min_child_samples": 30, "random_state": 0, },
            {"n_estimators": 25, "learning_rate": 0.2, "num_leaves": 31,
             "max_depth": 12, "min_child_samples": 30, "random_state": 0, },
            {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 31,
             "max_depth": 12, "min_child_samples": 30, "random_state": 0, }, ]
        objs = [("custom_revenue", {"f_revenue": cus_obj.custom_revenue,
                                    "y_transform": cus_obj.custom_revenu_transform}),
                ("custom_revenue2", {"f_revenue": cus_obj.custom_revenue2,
                                     "y_transform": cus_obj.custom_revenu2_transform}),
                ("l2", {"f_revenue": cus_obj.l2_revenue}),
                ("l2", {"f_revenue": cus_obj.l2_revenue})]
        targets = ["y_l_rise", "y_s_rise", "y_l_decline", "y_s_decline", "y_l_avg",
                   "y_s_avg", "y_l", "y_l_r"]
        #
        layer0 = {}
        for target in targets[:6]:
            layer0.update({obj_type + "_" + target: (
            lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target}) for
                           (obj_type, obj_dict), kwargs in
                           zip(objs[:2], reg_params[:2])})
        # del layer0["l2_y_l"]

        layer1 = {}
        for target in targets[:6]:
            layer1.update({obj_type + "_" + target: (
            lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target}) for
                           (obj_type, obj_dict), kwargs in
                           zip(objs[2:3], reg_params[2:3])})

        layer2 = {}
        for target in targets[-1:]:
            layer2.update({obj_type + "_" + target: (
            lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target}) for
                           (obj_type, obj_dict), kwargs in
                           zip(objs[-1:], reg_params[-1:])})
        layers = [layer0, layer1, layer2]

        lgbm_reg_net = ml.RegressorNetwork()
        lgbm_reg_net.insert_multiple_layers(layers)

        print("dataset range:",start,end)
        paras = {"fit": {"categorical_feature": cols_category}}
        for i in range(lgbm_reg_net.get_num_layers()):
            X, Y, _ = IO_op.read_hdf5(start=start, end=end,
                                      subsample="10-{0}".format(i))
            print(X.info(memory_usage="deep"))
            del X["industry"]

            Y["y_l_r"] = Y.apply(
                lambda r: r["y_l_rise"] if r["y_l_avg"] > 0 else r["y_l_decline"],
                axis=1) * 0.75 + 0.25 * Y["y_l_avg"].values
            print(Y[Y["y_l_avg"].isnull()].shape)
            print(Y[Y["y_l_avg"].isnull()].iloc[:20])

            cond = Y.notnull().all(axis=1)
            X = X[cond]
            Y = Y[cond]

            trading_dates = Y.index.unique().sort_values(ascending=True)
            train_dates = trading_dates[:-21]
            X = X.loc[train_dates]
            Y = Y.loc[train_dates]
            print("Train dates:{0}-{1}".format(min(train_dates), max(train_dates)))
            print(X.shape)

            if i > 0:
                for j in range(i):
                    features, _ = lgbm_reg_net.predict_layer(j, X)
                    X = pd.concat([X, features], axis=1)

            lgbm_reg_net.fit_layer(i, X, Y, **paras)
            del X, Y

        model_dir = "models"
        model_f_name = "lgbm_reg_net"+"_{0}-{1}".format(start,end)
        model_path = os.path.join(model_dir, model_f_name + "_{0}".format(
            datetime.datetime.now().strftime("%Y%m%d")))
        with open(model_path, mode="wb") as f:
            pickle.dump(lgbm_reg_net, f)

        print("test range:",test_start,test_end)
        X, Y, _ = IO_op.read_hdf5(start=test_start, end=test_end,
                                  # subsample="10-{0}".format(0)
                                  )
        del X["industry"]
        cond = Y.notnull().all(axis=1)
        X = X[cond]
        Y = Y[cond]

        result = lgbm_reg_net.predict(X)
        start_idx, end_idx = -6, 6
        col = "layer{0:d}_l2_y_l_r_pred".format(len(lgbm_reg_net.layers) - 1)

        print("\n" + ycol1)
        pred_interval_summary(lgbm_reg_net, X,
                              Y[ycol1],
                              y_test_pred=result[col])
        print("\n" + ycol2)
        pred_interval_summary(lgbm_reg_net, X,
                              Y[ycol2],
                              y_test_pred=result[col])
        print("\n" + ycol3)
        pred_interval_summary(lgbm_reg_net, X,
                              Y[ycol3],
                              y_test_pred=result[col])

        print("\n" + ycol4)
        pred_interval_summary(lgbm_reg_net, X,
                              Y[ycol4],
                              y_test_pred=result[col])
        del X,Y

