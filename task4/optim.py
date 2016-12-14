import numpy as np
import scipy as sp
from logutils import FuncSumWrapper, Logger
import random
from numpy.linalg import norm

def sgd(fsum, x0, n_iters=1000, step_size=0.1, trace=False):

    fsum = FuncSumWrapper(fsum)
    n_funcs = fsum.n_funcs
    x_out = x0
    x_cur = x0

    if trace:
        logger = Logger(fsum)

    for k in range(n_iters):
        i_cur = random.randint(0, n_funcs-1)
        x_cur = x_cur - step_size * fsum.call_ith(i_cur, x_cur)[1]
        x_out = x_out + x_cur
        logger.record_point(1/(k + 1) * x_out)

    x_out /= n_iters

    if trace:
        hist = logger.get_hist()
        return x_out, hist
    else :
        return x_out

def svrg(fsum, x0, n_stages=10, n_inner_iters=None, tol=1e-4, trace=False, L0=1):

    fsum = FuncSumWrapper(fsum)
    n_funcs = fsum.n_funcs
    x_s = x0
    L = L0

    if n_inner_iters == None :
        n_iter_inners = 2 * n_funcs

    if trace:
        logger = Logger(fsum)

    for s in range(n_stages) :
        g_s = fsum.call_ith(0,x_s)[1]
        for i in range(1, n_funcs):
            g_s = g_s + fsum.call_ith(i,x_s)[1]
        g_s = (1/n_funcs) * g_s
        x_k = x_s
        x_out_in = np.zeros_like(x_k)
        for k_in in range(n_iter_inners):
            i_cur = random.randint(0, n_funcs-1)
            g_k_in = fsum.call_ith(i_cur, x_k)[1] - fsum.call_ith(i_cur, x_s)[1] + g_s
            while True :
                x = x_k - (0.1/L)*g_k_in
                f_i_cur = fsum.call_ith(i_cur, x_k)
                nesterov_subject = fsum.call_ith(i_cur, x)[0] > f_i_cur[0] + f_i_cur[1].dot(x - x_k) + (L/2)*norm(x - x_k)**2
                if nesterov_subject :
                    L *= 2
                if not nesterov_subject :
                    break
            x_k = x
            x_out_in = x_out_in + x_k
            L = max(L0, L / (2**(1/n_iter_inners)))
            logger.record_point((1 / (k_in + 1)) * x_out_in)
        x_s = (1 / (n_iter_inners))*x_out_in

    if trace:
        hist = logger.get_hist()
        return x_s, hist
    else :
        x_s
