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


def get_return_rate(y_true):
    r = np.array(y_true)
    r[r <= -0.1] = -0.1
    r[r > -0.1] = r[r > -0.1]*0.7
    # print(r.shape)
    return r


def custom_revenue_obj(y_true,y_pred):
    r = get_return_rate(y_true)

    sigmoid = 1/(1+np.exp(-y_pred))
    grad = -r *sigmoid*(1-sigmoid)
    hess = np.ones(shape=y_true.shape)
    return grad,hess


def custom_revenue(y_true, y_pred):
    r = get_return_rate(y_true)
    y_pred = custom_revenu_transform(y_pred)

    revenue = r * y_pred
    return y_pred,r,revenue,sum(revenue)


def custom_revenu_transform(y_pred):
    return 1 /(1+np.exp(-y_pred))


def custom_revenue2_obj(y_true, y_pred):
    r = get_return_rate(y_true)

    sign = r.copy()
    sign[sign>0]=1
    sign[sign<0]=0

    y_pred = custom_revenu2_transform(y_pred)
    grad = -r *np.abs(sign-y_pred)
    hess = np.ones(shape=y_true.shape)
    return grad,hess


def custom_revenue2(y_true, y_pred):
    y_true = pd.Series(y_true)
    r = get_return_rate(y_true)
    y_pred = custom_revenu2_transform(y_pred)

    revenue = r * y_pred
    return y_pred,r,revenue,sum(revenue)


def custom_revenu2_transform(y_pred):
    y_pred[y_pred > 1] = 1
    y_pred[y_pred < 0] = 0
    return y_pred


def l2_revenue(y_true,y_pred):
    y_true = pd.Series(y_true)
    r = y_true.copy(deep=True)

    return y_pred,r,r,sum(r)


