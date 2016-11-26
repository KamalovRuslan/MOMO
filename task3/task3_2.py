import numpy as np
import scipy as sp
from numpy import linalg as la

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

def barrier(X, y, reg_coef, w0_plus, w0_minus, tol=1e-5, tol_inner=1e-7, max_iter=100, max_iter_inner=20, t0=1, gamma=10, c1=1e-4, disp=False, trace=False) :

    n = y.size
    A = X.T.dot(X) / n
    d = X.shape[1]
    y_ = X.T.dot(y) / n
    tau = t0

    def matvec(p,w_pl = 0, w_mi = 0, t = 1) :

        res = A.dot(p)
        res = np.array([(1 + (w_mi[i] ** 2) / (w_pl[i] ** 2)) * res[i] for i in range(res.size)])
        res = t * res + np.array([p / w ** 2 for p, w in zip(p,w_pl)])
        return res

    def generate_b(w_plus, w_minus, tau):
        b = w_plus - w_minus + 2 * tau *  reg_coef * np.array([w ** 2 for w in w_minus])
        b = b - np.array([w_m ** 2 / w_p for w_m, w_p in zip(w_minus, w_plus)]) - w_minus
        b = tau * (-A.dot(b) + y_) + np.array([1 / w - reg_coef * tau for w in w_plus])
        return b

    def select_alpha(w,p) :
        alpha_list = [-w_ / p_ for w_,p_ in zip(w,p) if p_ < 0]
        alpha = np.inf if not alpha_list else min(alpha_list)
        return alpha

    def func(w_plus, w_minus, tau) :
        ans = 0.5 * tau / n * la.norm(X.dot(w_plus - w_minus) - y) ** 2 + tau * reg_coef * sum(list(w_plus + w_minus))
        ans -= sum([np.log(w_p) + np.log(w_m) for w_p,w_m in zip(w_plus, w_minus)])
        return ans

    def armiho(w_plus, w_minus, p_plus, p_minus, f, dot, alpha, tau) :
        return func(w_plus + alpha * p_plus, w_minus + alpha * p_minus,tau) - c1 * alpha * dot - f < 0


    w_plus = w0_plus
    w_minus = w0_minus

    n_iter = 0
    status = 1

    while True :

        grad_plus = tau * A.dot(w_plus - w_minus) - tau * y_
        grad_plus = grad_plus + np.array([tau * reg_coef - 1 / w for w in w_plus])
        grad_minus = -tau * A.dot(w_plus - w_minus) + tau * y_
        grad_minus = grad_minus + np.array([tau * reg_coef - 1 / w for w in w_minus])
        norm_g = la.norm(np.array(list(grad_plus) + list(grad_minus)))
        n_iter_inner = 0

        while norm_g > tol_inner and n_iter_inner < max_iter_inner :

            b = generate_b(w_plus, w_minus, tau)
            p_plus, st = cg(lambda p : matvec(p, w_pl = w_plus, w_mi = w_minus, t = tau), b, np.zeros(d), tol = 1e-8)
            p_minus = np.array([w_m ** 2 * (-2 * reg_coef * tau + 1 / w_p) + w_m - (w_m ** 2) / (w_p ** 2) * p for w_m,w_p,p in zip(w_minus, w_plus, p_plus)])

            alpha = select_alpha(list(w_plus) + list(w_minus), list(p_plus) + list(p_minus))
            alpha = min(1, 0.95 * alpha)

            f = func(w_plus, w_minus, tau)
            dot = grad_plus.dot(p_plus) + grad_minus.dot(p_minus)

            while not armiho(w_plus, w_minus, p_plus, p_minus, f, dot, alpha, tau) :
                print(n_iter_inner)
                alpha /= 2

            w_plus = w_plus + alpha * p_plus
            w_minus = w_minus + alpha * p_minus

            grad_plus = tau * A.dot(w_plus - w_minus) - tau * y_
            grad_plus = grad_plus + np.array([tau * reg_coef - 1 / w for w in w_plus])
            grad_minus = -tau * A.dot(w_plus - w_minus) + tau * y_
            grad_minus = grad_minus + np.array([tau * reg_coef - 1 / w for w in w_minus])
            norm_g = la.norm(np.array(list(grad_plus) + list(grad_minus)))

            n_iter_inner += 1
        print(norm_g)

        mu = min(1, n * reg_coef / (la.norm(X.dot(w_plus - w_minus) - y, np.inf))) * (X.dot(w_plus - w_minus) - y) / n
        dual_gap = 0.5 / n * la.norm(X.dot(w_plus - w_minus) - y) ** 2 + reg_coef * la.norm(w_plus - w_minus, 1) + n / 2 * la.norm(mu) ** 2 + mu.dot(y)
        print(dual_gap)

        n_iter += 1

        if dual_gap < tol or  n_iter == max_iter :
            status = 0 if dual_gap < tol else 1
            break

        tau *= gamma
    print(tau)
    return w_plus - w_minus, status


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, -1, 1, 0])
reg_coef = 1
w0 = np.array([1, 1])
w0_plus = np.array([1, 1])
w0_minus = np.array([2, 2])

w,s = barrier(X, y, reg_coef, w0_plus, w0_minus, max_iter = 100, max_iter_inner = 6)
print(w)
print(s)
