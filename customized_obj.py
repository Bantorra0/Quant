import numpy as np


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

