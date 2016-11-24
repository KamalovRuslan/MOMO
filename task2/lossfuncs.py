import numpy as np
import scipy as sp
import scipy.sparse as ss
import math
from numpy import linalg as la
from scipy.misc import logsumexp
from functools import reduce
from scipy.sparse import csr_matrix

def logistic(w, X, y, reg_coef) :

    M = X.dot(w) * y
    g_M = list(map(lambda x: sp.exp(-logsumexp([0,x])),M))

    f = 1 / y.size * reduce(lambda res,x : res + logsumexp([0,x]), -M, 0) + 0.5 * reg_coef * w.dot(w)

    g = 1 / y.size * X.T.dot(-y * g_M) + reg_coef * w

    return f, g

def logistic_hess_vec(w, v, X, y, reg_coef):

    M = -X.dot(w) * y

    sigma = np.array(list(map(lambda x: sp.exp(-logsumexp([0,x])),M)))
    sigma = sigma * (1 - sigma)

    hv = 1  / y.size * X.T.dot(X.dot(v) * sigma) + reg_coef * v

    return hv
