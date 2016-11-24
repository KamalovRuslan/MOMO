import numpy as np
from numpy import linalg as la
import scipy
from scipy.optimize.linesearch import line_search_wolfe2
import time
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.misc import logsumexp
from functools import reduce

from lossfuncs import logistic, logistic_hess_vec

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize

def cg(matvec, b, x0, tol=1e-4, max_iter=None, disp=False, trace=False):
    if max_iter  == None :
        max_iter = x0.size

    x_sol = x0
    discrepency = matvec(x_sol) - b
    direction = -discrepency
    norm_dis = la.norm(discrepency)
    n_iter = 0

    norm_r = list()

    while n_iter < max_iter and norm_dis > tol :

        A_d = matvec(direction)

        alpha = norm_dis ** 2 / np.dot(direction, A_d)
        x_sol = x_sol + alpha * direction
        discrepency = discrepency + alpha * A_d

        beta = np.dot(discrepency, discrepency) / (norm_dis * norm_dis)
        direction = -discrepency + beta * direction

        norm_dis = la.norm(discrepency)
        n_iter = n_iter + 1

        norm_r.append(norm_dis)

        if disp:
            print('%s %3d %s %5f' % ('#:', n_iter,'   norm_r:',norm_dis))

    status = 0 if norm_dis < tol else 1

    if trace :
        hist = {'norm_r' : np.array(norm_r)}
        return x_sol, status, hist
    else :
        return x_sol,status



class FuncWrapper:
    def __init__(self, func):
        self.func = func
        self.x_last = None
        self.out_last = None
        self.n_counter = 0
    def __call__(self, x):
        if self.x_last is None or not np.all(self.x_last == x):
            self.out_last = self.func(x)
            self.x_last = np.copy(x)
            self.n_counter += 1
        return self.out_last



def hfn(func, x0, hess_vec, tol=1e-4, max_iter=500, c1=1e-4, c2=0.9, disp=False, trace=False):

    t_start = time.time()

    func_wrapper = FuncWrapper(func)
    func_f = lambda x: func_wrapper(x)[0]
    func_g = lambda x: func_wrapper(x)[1]

    x_min = x0
    f_min, g_min = func_f(x_min), func_g(x_min)
    n_iter = 0
    norm_g = la.norm(g_min, np.inf)

    list_f = list()
    list_norm_g = list()
    list_n_evals = list()
    list_elaps_t = list()


    while n_iter < max_iter and norm_g >= tol :
        hess_vec_c = lambda v: hess_vec(x_min, v)
        forcing = min(0.5, norm_g ** (0.5))
        cg_answer = cg(hess_vec_c, -g_min, x_min, tol=forcing * norm_g, trace = True)
        direction = cg_answer[0]
        pre_direction = direction
        not_direction = np.dot(direction, -g_min) <= 0
        while not_direction :
            forcing = 0.1 * forcing
            cg_answer = cg(hess_vec_c, -g_min, pre_direction, tol=forcing * norm_g, trace = True)
            direction = cg_answer[0]
            not_direction = np.dot(direction, -g_min) <= 0

        wolfe_answer = line_search_wolfe2(func_f, func_g, x_min, direction, func_g(x_min), c1=c1, c2=c2)
        alpha = wolfe_answer[0]

        x_min = x_min + alpha * direction
        f_min, g_min = func_f(x_min), func_g(x_min)
        norm_g = la.norm(g_min, np.inf)
        elaps_t = time.time() - t_start

        n_iter = n_iter + 1

        list_f.append(f_min)
        list_norm_g.append(norm_g)
        list_n_evals.append(func_wrapper.n_counter)
        list_elaps_t.append(elaps_t)

        if disp:
            print('%s %3d %s %5f %s %5f %s %3d %s %8f' % ('#:', n_iter,'   f:',f_min,'   norm_g:',norm_g,'   n_evals:',func_wrapper.n_counter,'   elaps_t:',elaps_t))

    status = 0 if norm_g < tol else 1

    if trace :
        hist = {'f' : np.array(list_f), 'norm_g' : np.array(list_norm_g), 'n_evals' : np.array(list_n_evals), 'elaps_t' : np.array(list_elaps_t)}
        return x_min, f_min, status, hist
    else :
        return x_min, f_min, status

def ncg(func, x0, tol=1e-4, max_iter=500, c1=1e-4, c2=0.1, disp=False, trace=False) :

    t_start = time.time()

    func_wrapper = FuncWrapper(func)
    func_f = lambda x: func_wrapper(x)[0]
    func_g = lambda x: func_wrapper(x)[1]

    x_min = x0
    f_min, g_min = func_f(x_min), func_g(x_min)
    direction = -g_min
    norm_g = la.norm(g_min, np.inf)
    n_iter = 0

    list_f = list()
    list_norm_g = list()
    list_n_evals = list()
    list_elaps_t = list()


    while n_iter < max_iter and norm_g > tol :

        wolfe_answer = line_search_wolfe2(func_f, func_g, x_min, direction, c1=c1, c2=c2)
        alpha = wolfe_answer[0]

        x_min = x_min + alpha * direction
        pre_g_min = g_min
        f_min, g_min = func_f(x_min), func_g(x_min)
        norm_g = la.norm(g_min, np.inf)
        elaps_t = time.time() - t_start

        beta = np.dot(g_min, g_min) / np.dot(direction, g_min - pre_g_min)
        direction = -g_min + beta * direction

        n_iter = n_iter + 1

        list_f.append(f_min)
        list_norm_g.append(norm_g)
        list_n_evals.append(func_wrapper.n_counter)
        list_elaps_t.append(elaps_t)

        if disp:
            print('%s %3d %s %5f %s %5f %s %3d %s %10f' % ('#:', n_iter,'   f:',f_min,'   norm_g:',norm_g,'   n_evals:',func_wrapper.n_counter,'   elaps_t:',elaps_t))

    status = 0 if norm_g < tol else 1

    if trace :
        hist = {'f' : np.array(list_f), 'norm_g' : np.array(list_norm_g), 'n_evals' : np.array(list_n_evals), 'elaps_t' : np.array(list_elaps_t)}
        return x_min, f_min, status, hist
    else :
        return x_min, f_min, status

def lbfgs_compute_dir(sy_hist, g) :

    d = -g

    if (len(sy_hist) == 0) :
        return d

    alpha_l = list()

    for i in range (len(sy_hist)) :
        s = sy_hist[-i - 1][0]
        y = sy_hist[-i - 1][1]
        alpha = np.dot(s, d) / (np.dot(s,y))
        alpha_l.append(alpha)
        d = d - alpha * y

    s = sy_hist[-1][0]
    y = sy_hist[-1][1]
    d = np.dot(s,y) / np.dot(y,y) * d

    for i in range(len(sy_hist), 0, -1) :
        s = sy_hist[-i][0]
        y = sy_hist[-i][1]
        beta = np.dot(y,d) / np.dot(s,y)
        d = d + (alpha_l[i-1] - beta) * s

    return d

def lbfgs(func, x0, tol=1e-4, max_iter=500, m = 10, c1=1e-4, c2=0.9, disp=False, trace=False) :

        t_start = time.time()

        func_wrapper = FuncWrapper(func)
        func_f = lambda x: func_wrapper(x)[0]
        func_g = lambda x: func_wrapper(x)[1]

        x_min = x0
        f_min, g_min = func_f(x_min), func_g(x_min)
        sy_hist = deque(maxlen = m)
        n_iter = 0
        norm_g = la.norm(g_min, np.inf)

        list_f = list()
        list_norm_g = list()
        list_n_evals = list()
        list_elaps_t = list()


        while n_iter < max_iter and norm_g > tol :

            pre_x = x_min
            pre_g = g_min

            direction = lbfgs_compute_dir(sy_hist, g_min)

            wolfe_answer = line_search_wolfe2(func_f, func_g, x_min, direction, g_min, c1=c1, c2=c2)
            alpha = wolfe_answer[0]

            x_min = x_min + alpha * direction
            f_min, g_min = func_f(x_min), func_g(x_min)
            norm_g = la.norm(g_min, np.inf)
            elaps_t = time.time() - t_start

            s = x_min - pre_x
            y = g_min - pre_g
            sy_hist.append(np.array([s,y]))

            n_iter = n_iter + 1

            list_f.append(f_min)
            list_norm_g.append(norm_g)
            list_n_evals.append(func_wrapper.n_counter)
            list_elaps_t.append(elaps_t)

            if disp:
                print('%s %3d %s %5f %s %5f %s %3d %s %10f' % ('#:', n_iter,'   f:',f_min,'   norm_g:',norm_g,'   n_evals:',func_wrapper.n_counter,'   elaps_t:',elaps_t))

        status = 0 if la.norm(g_min, np.inf) < tol else 1

        if trace :
            hist = {'f' : np.array(list_f), 'norm_g' : np.array(list_norm_g), 'n_evals' : np.array(list_n_evals), 'elaps_t' : np.array(list_elaps_t)}
            return x_min, f_min, status, hist
        else :
            return x_min, f_min, status
