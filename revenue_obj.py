import lightgbm as lgbm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def custom_revenue_obj(y_true,y_pred):
    y_true = pd.Series(y_true)
    r = y_true.copy(deep=True)
    idx = np.nonzero(y_true <= -0.1)
    r.iloc[idx] = -0.1
    idx = np.nonzero(y_true > -0.1)
    r.iloc[idx] = y_true.iloc[idx] * 0.7

    sigmoid = 1/(1+np.exp(-y_pred))
    grad = -r *sigmoid*(1-sigmoid)
    hess = np.ones(shape=y_true.shape)
    return grad,hess


def get_revenue(y_true,y_pred):
    y_true = pd.Series(y_true)
    r = y_true.copy(deep=True)
    idx = np.nonzero(y_true<=-0.1)
    r.iloc[idx] = -0.1
    idx = np.nonzero(y_true>-0.1)
    r.iloc[idx] = y_true.iloc[idx] * 0.7

    plt.figure()
    plt.hist(r)

    revenue = r /(1+np.exp(-y_pred))
    return r,revenue,sum(revenue)


def derive_feature(X):
    return X[:, 0] * np.exp(X[:, 1]) / (1 + X[:, 2] * X[:, 2]) + X[:, 3]


def filter(y0, a, b):
    return np.where((y0>a) & (y0<b),1,0)


if __name__ == '__main__':
    n_samples = 10000
    n_features = 10
    np.random.seed(0)
    X = np.random.normal(0,1,size=(n_samples,n_features))

    print(X.shape)
    print(X[:5])

    y0 = derive_feature(X)

    a1,b1 = 1,2
    a2,b2 = 3,4
    y1_1 = filter(y0,a1,b1)
    y1_2 = filter(y0,a2,b2)
    y2_1 = y1_1 * (y0+np.random.normal(0,0.1,size=n_samples)-(a1+b1)/2)*100
    y2_2 = y1_2 * (y0+np.random.normal(0,0.1,size=n_samples)-2.5)*10

    # print("Rule revenue:",sum(y2)*0.7)
    print("1-2",sum(y1_1!=0),sum(y1_1==0),y2_1.shape,sum(y2_1>10),sum(y2_1>20),sum(y2_1>30))
    print("3-4",sum(y1_2!=0),sum(y1_2==0),y2_2.shape,sum(y2_2>5),sum(y2_2>10),sum(y2_2>15))
    y1 = y1_1+y1_2
    y2 = y2_1 + y2_2
    y3 = (1-y1) * np.random.normal(-10,10,size=n_samples)/4 + (1-y1) * np.random.normal(-5,20,size=n_samples)*0.75
    print("y3==0:",sum(y3==0))

    t = (y3 + y2)/100

    plt.figure()
    plt.hist(y0,bins=np.arange(-20,21)*0.5)
    plt.figure()
    plt.hist(y2)
    plt.figure()
    plt.hist(y3)
    plt.figure()
    plt.hist(t,bins=np.arange(-70,70,5)/100)

    split_point = int(n_samples*0.7)
    X_train,X_test = X[:split_point],X[split_point:]
    t_train,t_test = t[:split_point],t[split_point:]
    print(X_train.shape,X_test.shape)
    print("feature count:",sum(y1),sum(y1[split_point:]))

    initial_learning_rate = 30
    reg1 = lgbm.LGBMRegressor(num_leaves=16, max_depth=6, n_estimators=400,
                              min_child_samples=5, objective=custom_revenue_obj,learning_rate=initial_learning_rate)
    decay_learning_rate = lambda n:initial_learning_rate/(1+n/50)
    f = lgbm.reset_parameter(learning_rate=decay_learning_rate)
    reg1.fit(X_train, t_train,callbacks=[f])


    y_pred1 = reg1.predict(X_test)
    sigmoid = pd.Series(1 / (1 + np.exp(-y_pred1)))
    plt.figure()
    plt.hist(sigmoid)
    idx = sigmoid.index[sigmoid>0.8]
    result1 = pd.DataFrame()
    result1["buy_pct"] = sigmoid
    result1["increase"] = t_test
    r,revenue,tot_revenue = get_revenue(t_test, y_pred1)
    result1["return_rate"] = r
    result1["revenue"] = revenue
    print(result1[result1["buy_pct"] > 0.8])


    print(reg1)
    print(tot_revenue,sum(r*0.5))
    for i in range(1,10):
        threshold = i*0.1
        print(">{0}:".format(threshold), result1[result1["buy_pct"] > threshold][
            "revenue"].sum(), sum(result1["buy_pct"] > threshold))

    reg2 = lgbm.LGBMRegressor(num_leaves=16, max_depth=6, n_estimators=400,
                              min_child_samples=5)
    reg2.fit(X_train,t_train)
    y_pred2 = reg2.predict(X_test)

    result2 = pd.DataFrame()
    result2["increase"] = t_test
    result2["pred"] = y_pred2
    result2["return_rate"] = r
    print("\n"+"-"*10+"\nPredict increase:")
    for i in range(0,11):
        threshold = i*0.05
        # print("\n>{0}:".format(threshold))
        # print(result2[result2["pred"]> threshold])
        print(">{0}:".format(threshold),result2[result2["pred"] > threshold]["return_rate"].sum(),sum(result2["pred"] > threshold))

    feature = derive_feature(X_test)
    filtered_feature = np.where(((feature > 1.5) & (feature < 2)), 1, 0) + np.where(((feature > 3) & (feature < 4)), 1, 0)
    print(sum(np.where(((feature > 1.5)), 1, 0)*result1["return_rate"]),
        sum(filtered_feature * result1["return_rate"]),
        sum(filtered_feature))

    plt.show()

