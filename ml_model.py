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

import db_operations as dbop
from data_prepare import prepare_data, gen_y, drop_null, feature_select, label

from constants import BASE_DIR


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


def gen_X(df_all: pd.DataFrame, cols_future, scaler=None, selector=None):
    features = df_all.columns.difference(cols_future + ["code"])
    X = df_all[features]
    if scaler:
        X = X.transform(X)
    if selector:
        X = X.transform(X)
    return X


def gen_dataset(pred_period=20, lower_bound="2011-01-01", start="2014-01-01",
                test_start="2018-01-01", label_type=None, is_drop_null=False,
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

    y = gen_y(df_all, threshold=0.15, pred_period=pred_period,
              label_type=label_type)
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


def train(data, models, is_reg=True):
    (X_train, y_train), (X_test, y_test)= data["train"],data["test"]
    if not is_reg:
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    y_pred_list = []
    colors = ["r", "b"]

    for model, c in zip(models, colors):
        t1 = time.time()
        model.fit(X_train, y_train)
        t2 = time.time()
        if is_reg:
            y_pred_list.append([model, t2 - t1, model.predict(X_test), c])
        else:
            y_pred_list.append(
                [model, t2 - t1, model.predict_proba(X_test), c])

    return y_pred_list


def main():
    data = gen_dataset(label_type=None)

    regs = [lgbm.LGBMRegressor(n_estimators=300, num_leaves=100, max_depth=8,
                               random_state=0),
            xgb.XGBRegressor(n_estimators=300, max_depth=5, random_state=0)]
    # _,(_,y_test),_ = data
    # plt.hist(y_test,bins=np.arange(-10,11)*0.1)
    y_pred_list_reg = train(data, regs, is_reg=True)

    clfs = [lgbm.LGBMClassifier(n_estimators=300, scale_pos_weight=0.1,
                                num_leaves=100, max_depth=8, random_state=0),
            xgb.XGBClassifier(n_estimators=300, scale_pos_weight=0.1,
                              max_depth=5, random_state=0, )]
    (X_train, y_train), (X_test, y_test)= data["train"],data["test"]
    y_train = label(y_train, 0.15, label_type="inc")
    y_test = label(y_test, 0.15, label_type="inc")
    data["train"],data["test"] = (X_train, y_train), (X_test, y_test)
    y_pred_list_clf = train(data, clfs, is_reg=False)

    X_test_full = data["full"][1]
    inc = X_test_full[["code", "f1mv_open", "f19max_f2mv_high"]].copy()
    inc["pct"] = inc["f19max_f2mv_high"] / inc["f1mv_open"] - 1
    y0 = inc["pct"].mean()
    print(y0)
    x0 = np.arange(11) * 0.1
    y0 = np.ones(x0.shape) * y0

    y_pred_clf = y_pred_list_clf[1][2][:, 1]
    threshold = 0.8
    clf_pred_inc = X_test_full[y_pred_clf > threshold][
        ["code", "f1mv_open", "f19max_f2mv_high"]]
    clf_pred_inc["pct"] = clf_pred_inc["f19max_f2mv_high"] / clf_pred_inc[
        "f1mv_open"] - 1
    clf_pred_inc["y_pred"] = y_pred_clf[y_pred_clf > threshold]
    # for code, group in clf_pred_inc.groupby("code"):
    #     print(group)

    y1 = []
    cnt1 = []
    for i in range(10):
        p0 = i * 0.1
        p1 = (i + 1) * 0.1
        cond = (p0 < y_pred_clf) & (y_pred_clf < p1)
        cnt1.append(sum(cond))
        y1.append(inc["pct"][cond].mean())
    for c, p in zip(cnt1, y1):
        print(c, p)
    print(sum([c * p for i, (c, p) in enumerate(zip(cnt1, y1)) if i > 6]))
    plt.figure()
    plt.bar(np.arange(len(y1)) * 0.1 + 0.05, y1, width=0.08)
    plt.plot(x0, y0, color='r')
    # plt.plot(x,y1,color='r')
    plt.xlim(0, 1)
    plt.ylim(0, 0.5)

    y_pred_reg = y_pred_list_reg[1][2]
    threshold = 0.3
    reg_pred_inc = X_test_full[y_pred_reg > threshold][
        ["code", "f1mv_open", "f19max_f2mv_high"]]
    reg_pred_inc["pct"] = reg_pred_inc["f19max_f2mv_high"] / reg_pred_inc[
        "f1mv_open"] - 1
    reg_pred_inc["y_pred"] = y_pred_reg[y_pred_reg > threshold]
    # for code, group in reg_pred_inc.groupby("code"):
    #     print(group)

    y2 = []
    cnt2 = []
    for i in range(10):
        p0 = i * 0.1
        p1 = (i + 1) * 0.1
        cond = (p0 < y_pred_reg) & (y_pred_reg < p1)
        cnt2.append(sum(cond))
        y2.append(inc["pct"][cond].mean())
    for c, p in zip(cnt2, y2):
        print(c, p)
    print(sum([c * p for i, (c, p) in enumerate(zip(cnt2, y2)) if
               i > 2 and not np.isnan(p)]))
    plt.figure()
    plt.bar(np.arange(len(y2)) * 0.1 + 0.05, y2, width=0.08)
    plt.plot(x0, y0, color='r')
    # plt.plot(x,y2,color='r')
    plt.xlim(0, 1)
    plt.ylim(0, 0.5)

    # save model

    for model in regs+clfs:
        fname = re.search("\.([^.]*)'", str(type(model))).group(1)
        print(fname)
        with open(os.path.join(BASE_DIR, fname), \
                  "wb") as f:
            pickle.dump(model, f)

    plt.show()


if __name__ == '__main__':
    main()
