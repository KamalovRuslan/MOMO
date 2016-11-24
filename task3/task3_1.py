import numpy as np
import scipy as sp
from numpy import linalg as la

def barrier(X, y, reg_coef, w0_plus, w0_minus, tol=1e-5, tol_inner=1e-8, max_iter=100, max_iter_inner=20, t0=1, gamma=10, c1=1e-4, disp=False, trace=False) :

    def select_alpha(w, p) :
        alpha_list = [-w_ / p_ for w_,p_ in zip(w,p) if p_ < 0]
        alpha = np.inf if not alpha_list else min(alpha_list)
        return alpha

    def func(w_plus, w_minus, tau) :
        #print(w_plus)
        #print(w_minus)
        ans = 0.5 * tau / n * la.norm(X.dot(w_plus - w_minus) - y) + tau * reg_coef * sum(list(w_plus + w_minus))
        ans -= sum([np.log(w_p) + np.log(w_m) for w_p,w_m in zip(w_plus, w_minus)])
        return ans

    w_plus = w0_plus
    w_minus = w0_minus
    tau = t0
    n = y.size
    d = w_plus.size

    A = 1  / n * X.T.dot(X)
    A_y = 1 / n * X.T.dot(y)

    n_iter = 0

    while True :

        n_iter_inner = 0

        grad_ = tau * A.dot(w_plus - w_minus) - tau * A_y + tau * reg_coef * np.ones(d)
        grad_plus = grad_ - np.array([1 / w for w in list(w_plus)])
        grad_minus = grad_ - np.array([1 / w for w in list(w_minus)])
        norm_g = la.norm(list(grad_plus) + list(grad_minus))

        while norm_g > tol_inner  and n_iter_inner < max_iter_inner:

            A_ = tau * np.array([A[:, i] * (1 + w_minus[i] ** 2 / w_plus[i] ** 2) for i in range(d)])
            for i in range(d) :
                A_[i,i] += 1 / w_plus[i] ** 2
            b_ = -grad_plus - tau * np.array([A[:, i] * w_minus[i] ** 2 for i in range(d)]).dot(2 * tau * reg_coef * np.ones(d) - np.array([1 / w for w in list(w_plus)]) \
                                                                                                                          - np.array([1 / w for w in list(w_minus)]))

            p_plus = la.solve(A_,b_)

            p_minus = 2 * tau * reg_coef * np.ones(d) - np.array([1 / w for w in list(w_plus)]) - np.array([1 / w for w in list(w_minus)])
            p_minus += np.array([p / w ** 2 for p,w in zip(p_plus, w_plus)])
            p_minus = np.array([-p * w ** 2 for p,w in zip(p_minus, w_minus)])

            #print(p_plus)
            #print(p_minus)
            print(grad_plus.dot(p_plus) + grad_minus.dot(p_minus))

            alpha = select_alpha(list(w_plus) + list(w_minus), list(p_plus) + list(p_minus))
            alpha = min(1, 0.95 * alpha)

            #print(alpha)

            func_ = func(w_plus, w_minus, tau)
            cur_func = func(w_plus + alpha * p_plus, w_minus + alpha * p_minus, tau)
            cur_func -= c1 * alpha * (grad_plus.dot(p_plus) + grad_minus.dot(p_minus))
            armiho = cur_func - func_ > 0

            while armiho :
                alpha /= 2
                cur_func = func(w_plus + alpha * p_plus, w_minus + alpha * p_minus, tau)
                cur_func += c1 * alpha * (grad_plus.dot(p_plus) + grad_minus.dot(p_minus))
                armiho = cur_func - func_ > 0

            #print(alpha)

            w_plus = w_plus + alpha * p_plus
            w_minus = w_minus + alpha * p_minus

            grad_ = tau * A.dot(w_plus - w_minus) - tau * A_y + tau * reg_coef * np.ones(d)
            grad_plus = grad_ - np.array([1 / w for w in list(w_plus)])
            grad_minus = grad_ - np.array([1 / w for w in list(w_minus)])
            norm_g = la.norm(list(grad_plus) + list(grad_minus))

            n_iter_inner += 1

        mu = min(1 , reg_coef / (la.norm(A.dot(w_plus - w_minus) - A_y, np.inf)))
        mu = mu * (X.dot(w_plus - w_minus) - y) / n
        #print(la.norm(X.T.dot(mu), np.inf))
        dual_gap = 0.5  / n * la.norm(X.dot(w_plus - w_minus) - y) ** 2 + reg_coef * la.norm(w_plus - w_minus, 1) + n / 2 * la.norm(mu) ** 2 + mu.dot(y)

        n_iter += 1

        if dual_gap <= tol or n_iter == max_iter :
            break

        tau *= gamma
        #print(n_iter)

    status = 0 if dual_gap <= tol else 1
    return w_plus - w_minus, status, dual_gap


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, -1, 1, 0])
reg_coef = 0.01
w0 = np.array([1, 1])
w0_plus = np.array([10, 10])
w0_minus = np.array([20, 20])

w, status, d = barrier(X, y, reg_coef, w0_plus, w0_minus, max_iter=5, max_iter_inner=1)
print(w)
print(status)
print(d)
