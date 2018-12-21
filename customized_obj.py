import numpy as np


def sigmoid_linear_transform(x, a,b):
    x1 = a*(x+b)
    sigmoid = 1/(1+np.exp(-x1))
    grad = a * sigmoid *(1-sigmoid)
    hess = a * (1-2*sigmoid) * grad
    return sigmoid, grad,hess


def sigmoid_lt_obj_generator(a,b):
    def sigmoid_lt_obj(y_true,y_pred):
        _, grad, hess = sigmoid_linear_transform(y_pred,a,b)
        return -y_true * grad, np.ones(y_true.shape)
    return sigmoid_lt_obj
