import os
import pickle
import re
import time

import lightgbm.sklearn as lgbm
import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc
import xgboost.sklearn as xgb
from matplotlib import pyplot as plt

import customized_obj as cus_obj
import db_operations as dbop
import io_operations as IO_op
from constants import MODEL_DIR
from data_prepare import prepare_data, feature_select


def gen_data(targets=None, start="2014-01-01", lowerbound="2011-01-01", end=None, upperbound=None,
             stock_pool=None):
    db_type = "sqlite3"
    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    df_all, cols_not_in_X, cols_category,enc = \
        prepare_data(cursor, targets=targets,
                     start=start, lowerbound=lowerbound,
                     end=end, upper_bound=upperbound,
                     stock_pool=stock_pool)

    df_all = df_all[df_all.index >= start]
    return df_all, cols_not_in_X, cols_category,enc


def y_distribution(y,plot=False):
    y = y.copy().dropna()
    # print distribution of y
    print("y<-0.5:", sum(y < -0.5))
    for i in range(-5, 5):
        tmp1 = ((i * 0.1) <= y)
        tmp2 = (y < ((i + 1) * 0.1))
        if len(tmp1) == 0 or len(tmp2) == 0:
            tmp = [False]
        else:
            tmp = tmp1 & tmp2
        print("{0:.2f}<=y<{1:.2f}:".format(i * 0.1, (i + 1) * 0.1), sum(tmp))
    print("y>0.5", sum(y > 0.5))
    print("mean:",y.mean(),"median:",y.median(),"std:",y.std())
    if plot:
        plt.figure()
        plt.hist(y, bins=np.arange(-10, 11) * 0.1)


def gen_y(df_all: pd.DataFrame, pred_period=10, threshold=0.1, is_high=True,
          is_clf=False, target_col = None):
    if target_col is None:
        target_col = get_target_col(pred_period, is_high)
    y = df_all[target_col] / df_all["f1mv_open"] - 1

    y_distribution(y)

    # print(y[y.isna() & (df_all["f1mv_high"] == df_all["f1mv_low"])])
    y[y.notnull() & (df_all["f1mv_high"] == df_all["f1mv_low"])] = 0
    print("过滤一字涨停项或停牌（最高价=最低价）：", sum(df_all["f1mv_high"] == df_all[
        "f1mv_low"]))

    y_distribution(y)

    if is_clf:
        return label(y, threshold=threshold, is_high=is_high)
    else:
        return y


def get_target_col(pred_period = 20,is_high = True):
    if is_high:
        target_col = "f{}max_f1mv_high".format(pred_period)
    else:
        target_col = "f{}min_f1mv_low".format(pred_period)
    return target_col


def label(y, threshold=0.1, is_high=True):
    y = y.copy()
    if not is_high:
        y = -y
    y[y > threshold] = 1
    y[y <= threshold] = 0
    return y


def drop_null(X, y):
    Xy = np.concatenate((np.array(X), np.array(y).reshape(-1, 1)), axis=1)
    Xy = pd.DataFrame(Xy, index=X.index).dropna()
    X = Xy.iloc[:, :-1].copy()
    y = Xy.iloc[:, -1].copy()
    return X, y


def gen_dataset(targets=None,pred_period=20,lower_bound="2011-01-01", start="2014-01-01",
                test_start="2018-01-01", is_high=True, is_clf=False, is_drop_null=False,
                is_normalized=False, is_feature_selected=False):
    """
    Generate training and testing data to be passed to train().
    :param pred_period:
    :param is_drop_null:
    :param is_normalized:
    :param is_feature_selected:
    :return:
    """
    df_all, cols_not_in_X = gen_data(targets, lower_bound, start)

    y = gen_y(df_all, threshold=0.15, pred_period=pred_period, is_high=is_high,
              is_clf=is_clf)
    print("null:", sum(y.isnull()))

    features = df_all.columns.difference(cols_not_in_X + ["code"])

    X_full = df_all[y.notnull()]
    X = X_full[features]
    y = y.dropna()
    if is_drop_null:
        X, y = drop_null(X, y)

    print("X_full,X,y:", X_full.shape, X.shape, y.shape)
    print("total positive", sum(y))

    test_period = (X.index >= test_start)
    X_train, y_train = X[~test_period], y[~test_period]
    X_test, y_test = X[test_period], y[test_period]
    print(X_test.shape, y_test.shape)
    print("test positive:", sum(y_test))

    X_train_full = X_full[~test_period]
    X_test_full = X_full[test_period]
    print(X_test_full.shape, X_test.shape)

    scaler = None
    if is_normalized:
        scaler = preproc.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    selector = None
    if is_feature_selected:
        X_train, selector = feature_select(X_train, y_train)
        X_test = selector.transform(X_test)

    return {"train":(X_train,y_train),
            "test":(X_test, y_test),
            "full": (X_train_full, X_test_full),
            "preproc":(scaler, selector)}


def gen_X(df_all: pd.DataFrame, cols_not_in_X, scaler=None, selector=None):
    features = df_all.columns.difference(cols_not_in_X + ["code"])
    X = df_all[features]
    if scaler:
        X = X.transform(X)
    if selector:
        X = X.transform(X)
    return X


def train(data, models, is_clf=False):
    X_train, y_train = data["train"]

    models_time = []

    for model in models:
        t1 = time.time()
        model.fit(X_train, y_train)
        t2 = time.time()
        models_time.append([model, t2 - t1])

        print("training time:", t2-t1)

    return models_time


def pred_vs_real(inc:pd.DataFrame, y_pred):
    x_min = -1

    # Average for all.
    y0 = inc["pct"].mean()
    std0 = inc["pct"].std()
    print("test data: mean={:.4f},std={:.4f}".format(y0, std0))
    x0 = np.arange(x_min,11) * 0.1

    # prediction performance
    df = pd.DataFrame(columns=["p0","range","cnt","min","mean","median","max","std"])
    df = df.set_index(["p0"])
    x_middle, x_interval = y0, std0
    for i in range(-5,10):
        p0 = x_middle + i * x_interval
        p1 = p0 + x_interval
        cond = (p0 < y_pred) & (y_pred < p1)
        df.loc[i] = ("{:.4f}-{:.4f}".format(p0,p1),
                  sum(cond),
                  inc["pct"][cond].min(),
                  inc["pct"][cond].mean(),
                  inc["pct"][cond].median(),
                  inc["pct"][cond].max(),
                  inc["pct"][cond].std())
        if i>2 and sum(cond)>0:
            plt.figure()
            plt.title(df.loc[i, "range"])
            plt.hist(inc["pct"][cond], bins=5)
    print(df)

    plt.figure()
    plt.title("real-pred")
    cond_plt = [True]*len(y_pred) # All True.
    plt.scatter(y_pred[cond_plt],inc["pct"][cond_plt])


    # for p0_pred, c, p_real,s in zip(p_pred,cnt, y,std):
    #     print("{0:.1f}-{1:.1f}:".format(p0_pred,p0_pred+0.1),c, p_real, s)
    print(sum([row["cnt"] * row["mean"] for i, row in df.iterrows()
               if i>2 and row["cnt"]>0]))
    plt.figure()
    plt.bar(np.array(list(map(float,df.index))) + 0.05, df["mean"], width=0.08)
    plt.plot(x0, np.ones(x0.shape) * y0, color='r')
    # plt.xlim(-0.2, 1)
    return y0,std0


def save_model(model, pred_period=20, is_high=True):
    suffix = "high" if is_high else "low"
    f_name = re.search("\.([^.]*)'", str(type(model))).group(1)
    f_name += "_{}".format(pred_period) + suffix
    print(f_name)
    with open(os.path.join(os.getcwd(),MODEL_DIR, f_name), "wb") as f:
        pickle.dump(model, f)


def load_model(model_type:str, pred_period=20, is_high=True):
    suffix = "high" if is_high else "low"
    f_name = model_type+"_{}".format(pred_period) + suffix
    print(f_name)
    with open(os.path.join(os.getcwd(),MODEL_DIR, f_name), "rb") as f:
        model = pickle.load(f)
    return model


def train_save(pred_period = 20,is_high = True, is_clf=False):

    data = gen_dataset(is_high=is_high,is_clf=is_clf,pred_period=pred_period)

    if is_clf:
        _, y_train=data["train"]
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    if not is_clf:
        models = [lgbm.LGBMRegressor(n_estimators=300, num_leaves=100,
                                     max_depth=8,random_state=0,
                                     min_child_weight=5),
                  xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                   random_state=0, min_child_weight=5)]
    else:
        models = [lgbm.LGBMClassifier(n_estimators=300, scale_pos_weight=0.1,
                                    num_leaves=100, max_depth=8, random_state=0),
                xgb.XGBClassifier(n_estimators=300, scale_pos_weight=0.1,
                                  max_depth=5, random_state=0, )]
    y_pred_list = train(data, models, is_clf=is_clf)

    # save model
    for model in models:
        save_model(model,pred_period,is_high)

    return y_pred_list


def load_test(pred_period = 20,is_high = True, is_clf=False):
    model_type = "XGBRegressor"

    model = load_model(model_type,pred_period,is_high)

    dataset = gen_dataset(
        lower_bound="2015-01-01",start="2018-01-01",pred_period=pred_period, is_high=is_high,is_clf=is_clf)
    X_test, y_test = dataset["test"]
    _, X_test_full = dataset["full"]

    target_col = get_target_col(pred_period,is_high)
    inc = X_test_full[["code", "f1mv_open", target_col]].copy()
    inc["pct"] = inc[target_col] / inc["f1mv_open"] - 1

    y_pred = model.predict(X_test)

    y0,std0 = pred_vs_real(inc,y_pred)

    plt.show()


def pred_interval_summary(reg, X_test, ss_eval:pd.Series, interval=0.05,
                          y_test_pred=None):
    if y_test_pred is None:
        y_test_pred = reg.predict(X_test)

    n = int(1 / interval)
    x0 = np.arange(n + 1) * interval
    y0 = np.ones(x0.shape) * ss_eval.mean()

    df = pd.DataFrame(columns=["count",
                               "eval_mean","eval_median","eval_std","eval_max","eval_min"])
    df.index.name="pred_range"
    for i in range(-n, n):
        p0 = i * interval
        p1 = (i + 1) * interval
        cond = (p0 < y_test_pred) & (y_test_pred <= p1)
        row = {"count": sum(cond),
               "eval_mean": ss_eval[cond].mean(),
               "eval_median": ss_eval[cond].median(),
               "eval_std": ss_eval[cond].std(),
               "eval_max": ss_eval[cond].max(),
               "eval_min":ss_eval[cond].min()}
        pred_range="({0:.2f},{1:.2f}]".format(p0, p1)
        df.loc[pred_range]=row
    df = df.astype({"count":int})
    # for c, p in zip(cnt, y1):
    #     print(c, p)
    pd.set_option("display.max_columns",10)
    print(df.round({col:3 for col in df.columns if col[:4]=="eval"}))

    plt.figure()
    plt.bar(np.arange(-n, n) * interval + interval / 2, df["eval_mean"],
            width=0.8 * interval)

    plt.plot(x0, y0, color='r')
    plt.xlim(-1, 1)
    plt.ylim(-0.5, 0.5)
    return df


def assess_by_revenue(Y_test, f_revenue, paras: dict, y_pred=None,reg=None,
                      X_test=None):

    result = pd.DataFrame()
    if y_pred is None and reg is not None and X_test is not None:
        y_pred = reg.predict(X_test)
    elif y_pred is None:
        raise ValueError("Cannot obtain y_pred, args missing!")

    result["y_pred"] = y_pred
    y_test = Y_test[paras["target"]].values
    result[paras["target"]] = y_test
    output, r, revenue, tot_revenue = f_revenue(y_test,y_pred)
    result[paras["output"]] = output
    result["return_rate"] = r.values
    result["revenue"] = revenue.values
    for col in Y_test.columns:
        result[col] = Y_test[col].values

    plt.figure()
    plt.hist(output)

    print("\n"+"-"*10+paras["reg_info"]+"-"*10)
    if reg is not None:
        print(reg)
    print("Model total revenue:",tot_revenue)
    print("Random total revenue",sum(r * 0.5))
    df = pd.DataFrame(
        columns=["revenue_sum","revenue_mean","revenue_median","revenue_max",
                 "revenue_min","revenue_std","count"])
    df.index.name = "{0}".format(paras["output"])
    for left,right in paras["intervals"]:
        cond = (result[paras["output"]] >= left) \
               & (result[paras["output"]] < right)
        df.loc["[{0:.2f}-{1:.2f}]:".format(left,right)]=\
            {"revenue_sum": result[cond]["revenue"].sum(),
             "revenue_mean": result[cond]["revenue"].mean(),
             "revenue_median": result[cond]["revenue"].median(),
             "revenue_max": result[cond]["revenue"].max(),
             "revenue_min": result[cond]["revenue"].min(),
             "revenue_std": result[cond]["revenue"].std(),
             "count": result[cond]["revenue"].count(),
        }
    df = df.astype({"count": int})
    pd.set_option("display.max_columns", 10)
    # pd.set_option("display.max_colwidth",500)
    print(df.round({col: 3 for col in df.columns if col[:4] == "eval"}))

    return df


def get_feature_importance(reg, features:list):
    feature_importance = [[features[i],importance] for i, importance in enumerate(reg.feature_importances_)]
    df = pd.DataFrame(feature_importance,columns=["feature","importance_raw"])
    tot = df["importance_raw"].sum()
    df["importance_percent"] = df["importance_raw"]/tot * 100
    return df.sort_values("importance_raw",ascending=False)


class RegressorNetwork:
    def __init__(self):
        self.layers = []
        self.is_trained = [False]

    def insert_layer(self,layer:list):
        self.layers.append(layer)
        self.is_trained.append(False)

    def insert_multiple_layers(self,layers):
        """
        Insert multiple layers into the network.

        :param layers: [{"name":(reg,paras)}]
        :return:
        """
        self.layers.extend(layers)
        self.is_trained.extend([False]*len(layers))

    def insert_reg(self,reg,paras,i):
        self.layers[i][paras["name"]] = (reg,paras)
        self.is_trained = self.is_trained[:i] \
                          + [False] * len(self.is_trained[i:])

    def get_num_layers(self):
        return len(self.layers)

    def remove_layer(self,i):
        self.layers.remove(self.layers[i])
        self.is_trained = self.is_trained[:i] \
                          + [False] * len(self.is_trained[i+1:])

    def remove_reg(self,i,name):
        del self.layers[i][name]
        self.is_trained = self.is_trained[:i+1] \
                          + [False] * len(self.is_trained[i+1:])

    def gen_features(self, reg, X, reg_prefix, reg_paras):
        features = pd.DataFrame(index=X.index)

        y_pred = reg.predict(X)
        features[reg_prefix + "_pred"] = y_pred

        if "y_transform" in reg_paras:
            output = reg_paras["y_transform"](y_pred)
            features[reg_prefix + "_output"] = output

        leaves = pd.DataFrame(reg.predict(X, pred_leaf=True),index=X.index)
        leaf_columns = [reg_prefix + "_tree{}_leaf".format(i) for i in
                        range(leaves.shape[1])]
        leaves.columns = leaf_columns
        features = pd.concat([features,leaves],axis=1)
        categorical_features = leaf_columns
        return features,categorical_features


    def fit_layer(self, i, X, Y, **paras):
        print("\n"+"-"*10+"Train layer {0:d}".format(i)+"-"*10)
        # features_list = []
        # paras_next_layer = copy.deepcopy(paras)
        # layer_prefix = "layer{0}".format
        if "train_indexes" in paras:
            train_slice = slice(*paras["train_indexes"])
            print(train_slice)
        else:
            train_slice = slice(0,None)

        for reg_name,(reg,reg_paras) in self.layers[i].items():
            print("\nTrain "+reg_name)
            target = reg_paras["target"]
            t0 = time.time()
            reg.fit(X.iloc[train_slice], Y[target].iloc[train_slice], **paras[
                "fit"])
            reg._fobj=None
            print("Time usage: {0:.2f}s".format(time.time()-t0))
            print(reg)

            df_feature_importance = get_feature_importance(reg, X.columns)
            print(df_feature_importance[
                      df_feature_importance["importance_raw"] > 0].round(
                {"importance_percent": 2}).iloc[:20])
            print("Among {0} features, {1} features are not used in the model"
                  .format(len(df_feature_importance),sum(df_feature_importance["importance_raw"] == 0)))
        self.is_trained[i] = True


    def fit(self, X, Y, **paras):
        print("\n" + "-" * 20 + "Train network" + "-" * 20)
        num_layers = self.get_num_layers()
        n = len(X)
        sample_size = n//num_layers
        split_points = list(range(0,num_layers*sample_size,sample_size))+[None]

        for i in range(num_layers):
            if i>0:
                features,feature_info = self.predict_layer(i-1,X)
                X = pd.concat([X,features],axis=1)
            paras["train_indexes"] = (split_points[i],split_points[i+1])
            print("\n", X.shape, Y.shape, paras)
            self.fit_layer(i, X, Y, **paras)


    def predict_layer(self,i,X,y=None, **paras):
        if self.is_trained[i] == False:
            raise ValueError("Layer {0} need to be trained "
                             "first.".format(i))

        print("\n" + "-" * 10 + "Layer {0:d} predicts".format(i) + "-" * 10)
        features_list = []
        layer_prefix = "layer{0}".format(i)
        categorical_features = []
        for reg_name, (reg, reg_paras) in self.layers[i].items():
            reg_prefix = layer_prefix + "_" + reg_name

            y_pred = reg.predict(X,**paras)

            if y is not None:
                _, _, _, tot_revenue = reg_paras["f_revenue"](y, y_pred)
                print(layer_prefix,reg_name,"tot_revenue:",tot_revenue)

            reg_features, cols = self.gen_features(reg, X,reg_prefix,reg_paras)
            features_list.append(reg_features)
            categorical_features.extend(cols)

        feature_info = {"categorical_feature":categorical_features}
        features = pd.concat(features_list,axis=1)
        return features,feature_info


    def predict(self, X, num_layers=None, **paras):
        print("\n" + "-" * 20 + "Predict" + "-" * 20)
        if num_layers is None:
            num_layers = len(self.layers)
        features = None
        for i in range(num_layers):
            print("layer",i)
            features,_ = self.predict_layer(i,X,**paras)
            if i<num_layers-1:
                X = pd.concat([X,features],axis=1)
        return features


if __name__ == '__main__':
    # train_save(pred_period=5, is_high=True, is_clf=False)
    # train_save(pred_period=5, is_high=False, is_clf=False)

    # load_test(pred_period=5, is_high=False, is_clf=False)
    # load_test(pred_period=5, is_high=True, is_clf=False)

    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 256)
    X, Y, _ = IO_op.read_hdf5(start="2013-01-01", end="2019-01-01",
                              subsample="10-0")
    print(X.info(memory_usage="deep"))
    del X["industry"]
    # Y["y_l_r"] = Y.apply(
    #     lambda r: (10*r["y_l_rise"]+0*r["y_l_avg"]) / 10
    #     if r["y_l_avg"] > 0 else ( 10*r["y_l_decline"]+0*r["y_l_avg"]) / 10,
    #     axis=1)
    Y["y_l_r"]= Y.apply(
        lambda r: r["y_l_rise"]
        if r["y_l_avg"] > 0 else r["y_l_decline"],
        axis=1)*0.75 + 0.25 * Y["y_l_avg"].values
    print(Y[Y["y_l_avg"].isnull()].shape)
    print(Y[Y["y_l_avg"].isnull()].iloc[:20])
    # ss = Y["y_l_r"] + 0.25 * Y["y_l_avg"].values
    # print("=====",ss.shape,Y["y_l_r"].shape)
    # print(Y[Y["y_l_r"]!=ss].iloc[:20])
    # Y["y_l_r"]=ss
    cond = Y.notnull().all(axis=1)
    X = X[cond]
    Y = Y[cond]

    cols_category = ["area", "market", "exchange", "is_hs"]
    dates= [20160701,20170101,20170701,20180101,20180701,20190101]
    # dates = [20180701,20190101]
    for test_start,test_end in zip(dates[:-1],dates[1:]):
        print("\ntest period:{0}-{1}".format(test_start,test_end))
        # test_start = 20180701
        trading_dates = Y.index.unique().sort_values(ascending=True)
        train_dates = trading_dates[trading_dates < test_start][:-21]
        test_dates = trading_dates[(trading_dates >= test_start) & (
                trading_dates<test_end)]

        # X_train = X.loc[train_dates]
        # Y_train = Y.loc[train_dates]
        # X_test = X.loc[test_dates]
        # Y_test = Y.loc[test_dates]
        # del X, Y

        ycol1, ycol2, ycol3,ycol4 = "y_l_r", "y_l", "y_l_avg","y_l_rise"
        print(X.loc[train_dates].shape, X.loc[test_dates].shape)
        lgbm_reg_net = RegressorNetwork()

        # regs=[
        #     lgbm.LGBMRegressor(n_estimators=10, learning_rate=2, num_leaves=15,
        #                        max_depth=8,
        #                        objective=cus_obj.custom_revenue_obj,
        #                        min_child_samples=30, random_state=0, ),
        #     lgbm.LGBMRegressor(n_estimators=10, learning_rate=2, num_leaves=15,
        #                        max_depth=8,
        #                        objective=cus_obj.custom_revenue2_obj,
        #                        min_child_samples=30, random_state=0, ),
        #     lgbm.LGBMRegressor(n_estimators=25, num_leaves=31, max_depth=12,
        #                        min_child_samples=30, random_state=0,
        #                        learning_rate=0.2),
        #     lgbm.LGBMRegressor(n_estimators=50, num_leaves=31, max_depth=12,
        #                        min_child_samples=30, random_state=0),
        # ]
        reg_params = [
            {"n_estimators": 10, "learning_rate": 2, "num_leaves": 15,
             "max_depth": 8,
             "objective": cus_obj.custom_revenue_obj,
             "min_child_samples": 30, "random_state": 0, },
            {"n_estimators": 10, "learning_rate": 2, "num_leaves": 15,
             "max_depth": 8,
             "objective": cus_obj.custom_revenue2_obj,
             "min_child_samples": 30, "random_state": 0, },
            {"n_estimators": 25, "learning_rate": 0.2, "num_leaves": 31,
             "max_depth": 12,
             "min_child_samples": 30, "random_state": 0, },
            {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 31,
             "max_depth": 12,
             "min_child_samples": 30, "random_state": 0, },
        ]
        objs = [("custom_revenue",
                 {"f_revenue":cus_obj.custom_revenue,
                  "y_transform":cus_obj.custom_revenu_transform}),
                ("custom_revenue2",
                 {"f_revenue": cus_obj.custom_revenue2,
                  "y_transform":cus_obj.custom_revenu2_transform}),
                ("l2",{"f_revenue": cus_obj.l2_revenue}),
                ("l2", {"f_revenue": cus_obj.l2_revenue})
                ]
        targets = ["y_l_rise", "y_s_rise",
                   "y_l_decline", "y_s_decline",
                   "y_l_avg","y_s_avg",
                   "y_l","y_l_r"]
        #
        layer0 = {}
        for target in targets[:6]:
            layer0.update(
                    {obj_type + "_" + target: (lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target})
                     for (obj_type, obj_dict), kwargs in
                     zip(objs[:2], reg_params[:2])})
        # del layer0["l2_y_l"]

        layer1 = {}
        for target in targets[:6]:
            layer1.update(
                {obj_type + "_" + target: (lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target})
                 for (obj_type, obj_dict), kwargs in zip(objs[2:3], reg_params[2:3])})

        layer2 = {}
        for target in targets[-1:]:
            layer2.update(
                {obj_type + "_" + target: (lgbm.LGBMRegressor(**kwargs), {**obj_dict, "target": target})
                 for (obj_type, obj_dict), kwargs in zip(objs[-1:], reg_params[-1:])})

        # layers = [layer0,layer1]
        layers = [layer0,layer1,layer2]
        # print(len(layer0),len(layer1))
        # print(layer0)
        # print(layer1)

        lgbm_reg_net.insert_multiple_layers(layers)

        paras = {"fit":{"categorical_feature":cols_category}}
        # plt.figure()
        # plt.hist(Y.loc[test_dates][ycol1], bins=np.arange(-10, 11) * 0.1)
        # plt.figure()
        # plt.hist(Y.loc[test_dates][ycol3],bins=np.arange(-10, 11) * 0.1)

        lgbm_reg_net.fit(X.loc[train_dates], Y.loc[train_dates][ycol2],
                         **paras)
        result = lgbm_reg_net.predict(X.loc[test_dates])
        start_idx, end_idx = -6,6
        # assess_paras = {"target":"y_l",
        #           "output": "y_l_pred",
        #           "reg_info": "l2, y_l",
        #           "intervals":
        #               list(zip(np.arange(start_idx,end_idx) * 0.1, np.arange(
        #                   start_idx+1, end_idx+1) * 0.1))}
        col = "layer{0:d}_l2_y_l_r_pred".format(len(lgbm_reg_net.layers)-1)
        # assess_by_revenue(y_pred=result[col], Y_test=Y_test,
        #                   f_revenue=cus_obj.l2_revenue, paras=assess_paras)

        print("\n" + ycol1)
        pred_interval_summary(lgbm_reg_net, X.loc[test_dates], Y.loc[test_dates][
            ycol1], y_test_pred=result[col])
        print("\n"+ycol2)
        pred_interval_summary(lgbm_reg_net, X.loc[test_dates], Y.loc[test_dates][
            ycol2], y_test_pred=result[col])
        print("\n"+ycol3)
        pred_interval_summary(lgbm_reg_net, X.loc[test_dates], Y.loc[test_dates][ycol3],y_test_pred=result[col])

        print("\n" + ycol4)
        pred_interval_summary(lgbm_reg_net, X.loc[test_dates], Y.loc[test_dates][ycol4], y_test_pred=result[col])

    plt.show()