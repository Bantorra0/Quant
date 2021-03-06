import os
import pickle
import re
import time

import lightgbm.sklearn as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc
import xgboost.sklearn as xgb
from matplotlib import pyplot as plt

import db_operations as dbop
from data_prepare import prepare_data, feature_select

from constants import FLOAT_DELTA


def gen_data(pred_period=20, lower_bound="2011-01-01", start="2014-01-01"):
    db_type = "sqlite3"
    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    df_all, cols_future = prepare_data(cursor, pred_period=pred_period,
                                       start=lower_bound)

    data_period = (df_all.index >= start)
    df_all = df_all[data_period]

    df_all = df_all[df_all["amt"] != 0]
    return df_all, cols_future


def y_distribution(y):
    y = y.copy().dropna()
    # print distribution of y
    print("before", sum(y < 0))
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
    print("after", sum(y < 0))
    plt.figure()
    plt.hist(y, bins=np.arange(-10, 11) * 0.1)


def gen_y(df_all: pd.DataFrame, pred_period=10, threshold=0.1, is_high=True,
          is_clf=False):
    target_col = get_target_col(pred_period, is_high)
    y = df_all[target_col] / df_all["f1mv_open"] - 1

    y_distribution(y)

    # print(y[y.isna() & (df_all["f1mv_high"] == df_all["f1mv_low"])])
    y[y.notnull() & (df_all["f1mv_high"] == df_all["f1mv_low"])] = 0
    print("过滤涨停项：", sum(df_all["f1mv_high"] == df_all["f1mv_low"]))

    return label(y, threshold=threshold, is_high=is_high,is_clf=is_clf)


def get_target_col(pred_period = 20,is_high = True):
    if is_high:
        target_col = "f{}max_f2mv_high".format(pred_period-1)
    else:
        target_col = "f{}min_f1mv_low".format(pred_period)
    return target_col


def label(y, threshold=0.1, is_high=True, is_clf=False):
    if is_clf:
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


def gen_dataset(pred_period=20, lower_bound="2011-01-01", start="2014-01-01",
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
    df_all, cols_future = gen_data(pred_period, lower_bound, start)

    y = gen_y(df_all, threshold=0.15, pred_period=pred_period, is_high=is_high,
              is_clf=is_clf)
    print("null:", sum(y.isnull()))

    features = df_all.columns.difference(cols_future + ["code"])

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


def gen_X(df_all: pd.DataFrame, cols_future, scaler=None, selector=None):
    features = df_all.columns.difference(cols_future + ["code"])
    X = df_all[features]
    if scaler:
        X = X.transform(X)
    if selector:
        X = X.transform(X)
    return X


def train(data, models, is_clf=False):
    X_train, y_train = data["train"]


    y_pred_list = []

    for model in models:
        t1 = time.time()
        model.fit(X_train, y_train)
        t2 = time.time()
        y_pred_list.append([model, t2 - t1])

        print("training time:", t2-t1)

    return y_pred_list


def pred_vs_real(inc:pd.DataFrame, y_pred):
    x_min = -1

    # Average for all.
    y0 = inc["pct"].mean()
    print(y0)
    x0 = np.arange(x_min,11) * 0.1
    y0 = np.ones(x0.shape) * y0

    # prediction performance
    df = pd.DataFrame(columns=["p0","range","cnt","min","mean","median","max","std"])
    df = df.set_index(["p0"])
    for i in range(-5,10):
        p0 = i * 0.1
        p1 = (i + 1) * 0.1
        cond = (p0 < y_pred) & (y_pred < p1)
        df.loc["{:.1f}".format(p0)] = ("{:.1f}-{:.1f}".format(p0,p1),
                  sum(cond),
                  inc["pct"][cond].min(),
                  inc["pct"][cond].mean(),
                  inc["pct"][cond].median(),
                  inc["pct"][cond].max(),
                  inc["pct"][cond].std())
        if p0 > 0.3*FLOAT_DELTA and sum(cond)>0:
            plt.figure()
            plt.title(df.loc["{:.1f}".format(p0), "range"])
            plt.hist(inc["pct"][cond], bins=5)
    print(df)

    plt.figure()
    plt.title("real-pred")
    cond_plt = y_pred<0.5*FLOAT_DELTA
    plt.scatter(y_pred[cond_plt],inc["pct"][cond_plt])


    # for p0_pred, c, p_real,s in zip(p_pred,cnt, y,std):
    #     print("{0:.1f}-{1:.1f}:".format(p0_pred,p0_pred+0.1),c, p_real, s)
    print(sum([row["cnt"] * row["mean"] for p0, row in df.iterrows()
               if float(p0) < 0.3*FLOAT_DELTA and row["cnt"]>0]))
    plt.figure()
    plt.bar(np.array(list(map(float,df.index))) + 0.05, df["mean"], width=0.08)
    plt.plot(x0, y0, color='r')
    # plt.xlim(-0.2, 1)


def save_model(model, pred_period=20, is_high=True):
    suffix = "high" if is_high else "low"
    f_name = re.search("\.([^.]*)'", str(type(model))).group(1)
    f_name += "_{}".format(pred_period) + suffix
    print(f_name)
    with open(os.path.join(os.getcwd(), f_name), "wb") as f:
        pickle.dump(model, f)


def load_model(model_type:str, pred_period=20, is_high=True):
    suffix = "high" if is_high else "low"
    f_name = model_type+"_{}".format(pred_period) + suffix
    print(f_name)
    with open(os.path.join(os.getcwd(), f_name), "rb") as f:
        model = pickle.load(f)
    return model


def train_save(pred_period = 20,is_high = True, is_clf=False):

    data = gen_dataset(is_high=is_high,is_clf=is_clf,pred_period=pred_period)

    if is_clf:
        _, y_train=data["train"]
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    if not is_clf:
        models = [lgbm.LGBMRegressor(n_estimators=300, num_leaves=100, max_depth=8,random_state=0),
                  xgb.XGBRegressor(n_estimators=300, max_depth=5, random_state=0)]
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

    pred_vs_real(inc,y_pred)

    plt.show()


if __name__ == '__main__':
    # train_save(pred_period=5, is_high=True, is_clf=False)
    # train_save(pred_period=5, is_high=False, is_clf=False)

    # load_test(pred_period=5, is_high=False, is_clf=False)
    load_test(pred_period=5, is_high=True, is_clf=False)