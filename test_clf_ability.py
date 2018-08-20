import xgboost.sklearn as xgb
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import time
import sklearn.linear_model as lm
import scipy.sparse as sparse
import sklearn.neural_network as nn
import lightgbm.sklearn as lgbm
import numpy.random as rand


def make_X(n=100):
    m = n + 1
    x = np.arange(m).reshape(-1, 1) * np.ones(m) / n
    y = x.T
    x = np.ravel(x)
    y = np.ravel(y)
    X = np.array([x, y]).T
    return X


def make_target(X):
    x, y = X[:, 0], X[:, 1]
    labels = np.where(((x * x < y) & (x + y < 1)) | ((x * x >= y) & (x + y >= 1)), 1, 0)
    n = len(labels)
    idx = np.random.choice(n,size=n//10,replace=False)
    labels[idx[:len(idx)//2]]=0
    labels[idx[len(idx)//2:]]=1
    return labels


def real_labels(X, target):
    plt.figure()
    plt.scatter(*np.hsplit(X[target == 1], 2), s=5, c="r")
    plt.scatter(*np.hsplit(X[target == 0], 2), s=5, c="b")


def prd_labels(X, target, clf):
    X_train, X_test, t_train, t_test = sk.model_selection.train_test_split(X, target, train_size=0.7, random_state=0)

    t0 = time.time()
    clf.fit(X_train, t_train)
    t1 = time.time()
    t_prd = clf.predict(X)

    accuracy = metrics.accuracy_score(target, t_prd)
    print(clf.__class__, "\n", accuracy, t1 - t0, "\n")
    plt.figure()
    plt.title(clf.__class__)
    plt.scatter(*np.hsplit(X[t_prd == 1], 2), s=5, c="r")
    plt.scatter(*np.hsplit(X[t_prd == 0], 2), s=5, c="b")

    # if isinstance(clf, xgb.XGBClassifier) and clf.n_estimators == 10:
    #     lr1 = lm.LogisticRegression()
    #     enc = sk.preprocessing.OneHotEncoder()
    #     X_train_gbt = enc.fit_transform(clf.apply(X_train))
    #     t2 = time.time()
    #     lr1.fit(X_train_gbt, t_train)
    #     t3 = time.time()
    #     t_prd = lr1.predict(enc.transform(clf.apply(X)))
    #     accuracy = metrics.accuracy_score(target, t_prd)
    #     print(str(lr1), "\n", accuracy, t3 - t2, "\n")
    #     plt.figure()
    #     plt.title(str(lr1))
    #     plt.scatter(*np.hsplit(X[t_prd == 1], 2), s=5, c="r")
    #     plt.scatter(*np.hsplit(X[t_prd == 0], 2), s=5, c="b")
    #
    #     lr2 = lm.LogisticRegression()
    #     enc2 = sk.preprocessing.OneHotEncoder()
    #     X_train_gbt = enc2.fit_transform(clf.apply(X_train), t_train)
    #     # print(np.concatenate((X_train_gbt,X_train),axis=1))
    #     t2 = time.time()
    #     lr2.fit(sparse.hstack((X_train_gbt, X_train)), t_train)
    #     t3 = time.time()
    #     t_prd = lr2.predict(sparse.hstack([enc2.transform(clf.apply(X)), X]))
    #     accuracy = metrics.accuracy_score(target, t_prd)
    #     print(str(lr2), "\n", accuracy, t3 - t2, "\n")
    #     plt.figure()
    #     plt.title(str(lr2))
    #     plt.scatter(*np.hsplit(X[t_prd == 1], 2), s=5, c="r")
    #     plt.scatter(*np.hsplit(X[t_prd == 0], 2), s=5, c="b")
    #
    #     #
    #     lr3 = lm.LogisticRegression()
    #     enc3 = sk.preprocessing.OneHotEncoder()
    #     X_train_gbt = enc3.fit_transform(clf.apply(X_train))
    #     clf2 = xgb.XGBClassifier(n_estimators=5)
    #     t4 = time.time()
    #     clf2.fit(X_train_gbt, t_train)
    #     t5 = time.time()
    #     enc4 = sk.preprocessing.OneHotEncoder()
    #     X_train_gbt2 = enc4.fit_transform(clf2.apply(X_train_gbt))
    #     lr3.fit(X_train_gbt2, t_train)
    #     t_prd = lr3.predict(enc4.transform(clf2.apply(enc3.transform(clf.apply(X)))))
    #     accuracy = metrics.accuracy_score(target, t_prd)
    #     print(str(lr3), "\n", accuracy, t5 - t4, "\n")
    #     plt.figure()
    #     plt.title(str(lr3))
    #     plt.scatter(*np.hsplit(X[t_prd == 1], 2), s=5, c="r")
    #     plt.scatter(*np.hsplit(X[t_prd == 0], 2), s=5, c="b")


def main():
    X = make_X(200)
    target = make_target(X)

    real_labels(X, target)

    clf_list = [
        # nn.MLPClassifier(hidden_layer_sizes=(2,), random_state=0),
        # nn.MLPClassifier(hidden_layer_sizes=(3,), random_state=0),
        # nn.MLPClassifier(hidden_layer_sizes=(4,), random_state=0),
        # nn.MLPClassifier(hidden_layer_sizes=(10,), random_state=0),
        lgbm.LGBMClassifier(n_estimators=200,random_state=0),
        xgb.XGBClassifier(n_estimators=200,max_depth=5,random_state=0)

        # nn.MLPClassifier(hidden_layer_sizes=(200,)),
        # nn.MLPClassifier(hidden_layer_sizes=(300,)),
        # nn.MLPClassifier(hidden_layer_sizes=(200, 100)),
        # xgb.XGBClassifier(n_estimators=30, max_depth=3),
        # xgb.XGBClassifier(n_estimators=5, max_depth=3),
        # ensemble.AdaBoostClassifier(n_estimators=30, random_state=0)
    ]
    for clf in clf_list:
        prd_labels(X, target, clf)

    plt.show()


if __name__ == '__main__':
    main()
