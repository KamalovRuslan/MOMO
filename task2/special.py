import numpy as np
from numpy import linalg as la
import scipy as sp
from scipy.misc import logsumexp
from functools import reduce
import random
import matplotlib.pyplot as plt

def grad_finite_diff(func, x, eps=1e-8) :

    g = np.zeros(x.size)
    f_x = func(x)

    for i in range(x.size) :
        ort = np.zeros(x.size)
        ort[i] = 1
        g[i] = (func(x + eps * ort) - f_x) / eps

    return g


def hess_vec_finite_diff(func, x, v, eps=1e-5):

    hv = np.zeros(x.size)
    f_x = func(x)

    for i in range(x.size) :
        ort = np.zeros(x.size)
        ort[i] = 1
        hv[i] = (func(x + eps * v + eps * ort) - func(x + eps * v) - func(x + eps * ort) + f_x) / eps ** 2

    return hv
