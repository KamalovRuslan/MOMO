import numpy as np
from scipy.special import expit


def loss_func(x, A, b, reg_coef):
    z = -b * (A.dot(x))
    s = expit(z)
    f = (1/A.shape[0]) * np.sum(np.logaddexp(np.zeros_like(z), z)) + (reg_coef/2)*x.dot(x)
    g = (1/A.shape[0]) * (A.T.dot(-b * s)) + reg_coef*x
    return f, g


def predict_labels(A, x):
    ax = A.dot(x)
    b = np.ones(A.shape[0])
    b[ax < 0] = -1
    return b


class LossFuncSum:
    def __init__(self, A, b, reg_coef):
        self.A = A
        self.b = b
        self.reg_coef = reg_coef

        self.n_funcs = A.shape[0]

    def call_ith(self, i, x):
        return loss_func(x, self.A[[i], :], self.b[[i]], self.reg_coef)
