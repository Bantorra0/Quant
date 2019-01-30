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

import db_operations as dbop
from data_prepare import prepare_data, feature_select

from constants import FLOAT_DELTA, MODEL_DIR


def gen_data(targets=None, lower_bound="2011-01-01", start="2014-01-01",
             stock_pool=None):
    db_type = "sqlite3"
    conn = dbop.connect_db(db_type)
    cursor = conn.cursor()

    df_all, cols_not_in_X, cols_category,enc = \
        prepare_data(cursor, targets=targets,start=start,lowerbound=lower_bound, stock_pool=stock_pool)

    df_all = df_all[df_all.index >= start]
    return df_all, cols_not_in_X, cols_category,enc


def y_distribution(y):
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
    plt.figure()
    plt.hist(y, bins=np.arange(-10, 11) * 0.1)


def gen_y(df_all: pd.DataFrame, pred_period=10, threshold=0.1, is_high=True,
          is_clf=False):
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


def add_suffix_to_file_names(files:dict, suffix:str):
    """
    Add date suffix to given file names.

    :param files: A dict of file names.
    :param suffix:
    :return: A dict of file names with date suffix.
    """
    files = files.copy()
    for k in files.keys():
        f_name = files[k]
        if '.' in f_name:
            idx = f_name.rindex(".")
        else:
            idx = len(f_name)
        files[k] = (f_name[:idx]+"_{0}"+f_name[idx:]).format(suffix)
    return files


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


def pred_interval_summary(reg, X_test, ss_eval:pd.Series, interval=0.05):
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


def get_feature_importance(reg, features:list):
    feature_importance = [[features[i],importance] for i, importance in enumerate(reg.feature_importances_)]
    df = pd.DataFrame(feature_importance,columns=["feature","importance_raw"])
    tot = df["importance_raw"].sum()
    df["importance_percent"] = df["importance_raw"]/tot * 100
    return df.sort_values("importance_raw",ascending=False)


if __name__ == '__main__':
    # train_save(pred_period=5, is_high=True, is_clf=False)
    # train_save(pred_period=5, is_high=False, is_clf=False)

    # load_test(pred_period=5, is_high=False, is_clf=False)
    load_test(pred_period=5, is_high=True, is_clf=False)