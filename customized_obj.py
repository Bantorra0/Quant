import numpy as np
import pandas as pd


def sigmoid_linear_transform(x, a,b):
    x1 = a*(x+b)
    sigmoid = 1/(1+np.exp(-x1))
    grad = a * sigmoid *(1-sigmoid)
    hess = a * (1-2*sigmoid) * grad
    return sigmoid, grad,hess


def sigmoid_lt_obj_generator(a,b):
    """
    Return the customized objective using sigmoid and linear transformation.

    :param a:
    :param b:
    :return:
    """
    def sigmoid_lt_obj(y_true,y_pred):
        _, grad, hess = sigmoid_linear_transform(y_pred,a,b)
        return -y_true * grad, -y_true * hess
    return sigmoid_lt_obj


def smooth_abs(x:np.array, k):
    """
    Smooth version of abs() function.

    :param x:
    :return:
    """
    v = np.log(np.exp(k*x)+np.exp(-k*x))/k
    grad = np.tanh(k * x)
    tmp = (np.exp(k * x)+np.exp(-k * x))
    hess = 4 * k/ (tmp * tmp)
    return v, grad, hess


def smooth_l1(y_true:np.array, y_pred:np.array, k):
    """
    Smooth version of l1 loss.

    :param x:
    :return:
    """
    x = y_pred-y_true
    grad = np.tanh(k * x)
    tmp = (np.exp(k * x) + np.exp(-k * x))
    hess = 4 * k / (tmp * tmp)
    return grad, hess


def smooth_l1_obj_generator(k):
    def smooth_l1_obj(y_true,y_pred):
        return smooth_l1(y_true,y_pred,k)
    return smooth_l1_obj


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

    revenue = r /(1+np.exp(-y_pred))
    return r,revenue,sum(revenue)

