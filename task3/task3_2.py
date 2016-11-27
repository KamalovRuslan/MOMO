import numpy as np
import scipy as sp
from numpy import linalg as la

def barrier(X, y, reg_coef, w0_plus, w0_minus, tol=1e-5, tol_inner=1e-7, max_iter=100, max_iter_inner=20, t0=1, gamma=10, c1=1e-4, disp=False, trace=False) :

    n = y.size
    A = X.T.dot(X) / n
    d = X.shape[1]
    y_ = X.T.dot(y) / n
    tau = t0

    def matr(A,w_plus,w_minus) :
        d = w_plus.size
        res = np.array([A[:,i] * (1 + (w_minus[i] / w_plus[i]) ** 2) for i in range(d)]).T
        for i in range(d) :
            res[i][i] += (1 / w_plus[i] ** 2)
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
            p_plus = la.solve(matr(tau * A,w_plus,w_minus), b)
            p_minus = np.array([w_m ** 2 * (-2 * reg_coef * tau + 1 / w_p) + w_m - (w_m ** 2) / (w_p ** 2) * p for w_m,w_p,p in zip(w_minus, w_plus, p_plus)])
            dot = grad_plus.dot(p_plus) + grad_minus.dot(p_minus)

            alpha = select_alpha(list(w_plus) + list(w_minus), list(p_plus) + list(p_minus))
            alpha = min(1, 0.95 * alpha)

            f = func(w_plus, w_minus, tau)

            while not armiho(w_plus, w_minus, p_plus, p_minus, f, dot, alpha, tau) :
                alpha /= 2

            w_plus = w_plus + alpha * p_plus
            w_minus = w_minus + alpha * p_minus

            grad_plus = tau * A.dot(w_plus - w_minus) - tau * y_
            grad_plus = grad_plus + np.array([tau * reg_coef - 1 / w for w in w_plus])
            grad_minus = -tau * A.dot(w_plus - w_minus) + tau * y_
            grad_minus = grad_minus + np.array([tau * reg_coef - 1 / w for w in w_minus])
            norm_g = la.norm(np.array(list(grad_plus) + list(grad_minus)))

            n_iter_inner += 1

        mu = min(1, n * reg_coef / (la.norm(X.dot(w_plus - w_minus) - y, np.inf))) * (X.dot(w_plus - w_minus) - y) / n
        dual_gap = 0.5 / n * la.norm(X.dot(w_plus - w_minus) - y) ** 2 + reg_coef * la.norm(w_plus - w_minus, 1) + n / 2 * la.norm(mu) ** 2 + mu.dot(y)

        n_iter += 1

        if dual_gap < tol or  n_iter == max_iter :
            status = 0 if dual_gap < tol else 1
            break

        tau *= gamma
    return w_plus - w_minus, status


def subgrad(X, y, reg_coef, w0, tol=1e-2, max_iter=1000, alpha=1,disp=False, trace=False) :

    def subg(w) :
        return np.array([np.random.uniform(-1,1) if w[i] == 0 else np.sign(w[i]) for i in range(w.size)])

    def func(w) :
        return 0.5 / n * (la.norm(X.dot(w) - y) ** 2) + reg_coef * la.norm(w, 1)

    n = y.size
    A = 1 / n * X.T.dot(X)
    y_ = 1 / n * X.T.dot(y)

    w = w0
    f = func(w)
    direction_ = A.dot(w) - y_

    mu = min(1 , reg_coef / la.norm(A.dot(w) - y_), np.inf) * (X.dot(w) - y) / n
    dual_gap = 0.5 / n * la.norm(X.dot(w) - y) ** 2 + reg_coef * la.norm(w, 1) + 0.5 * n * la.norm(mu) ** 2 + mu.dot(y)

    n_iter = 0

    while dual_gap > tol and n_iter < max_iter :

        w_pred = w
        f_pred = f

        direction = reg_coef * subg(w) + direction_
        direction /= la.norm(direction)
        w = w - alpha / (n_iter + 1) ** 0.5 * direction
        f = func(w)

        if f_pred > f :
            mu = min(1 , reg_coef / la.norm(A.dot(w) - y_), np.inf) * (X.dot(w) - y) / n
            dual_gap = 0.5 / n * la.norm(X.dot(w) - y) ** 2 + reg_coef * la.norm(w, 1) + 0.5 * n * la.norm(mu) ** 2 + mu.dot(y)
            direction_ = A.dot(w) - y_
        else :
            w = w_pred
            f = f_pred

        n_iter += 1

    status = 0 if dual_gap < tol else 1
    return w, status



def prox_grad(X, y, reg_coef, w0, tol=1e-5, max_iter=1000, L0=1, disp=False, trace=False) :

    def prox(x, alpha) :
        ans = [x[i] + alpha if x[i] < -alpha else x[i] - alpha if x[i] > alpha else 0 for i in range(x.size)]
        return np.array(ans)

    def func(w) :
        return 0.5 / n * la.norm(X.dot(w) - y) ** 2 + reg_coef * la.norm(w,1)

    def m_L(y , x, grad, L) :
         return func(x) + grad.dot(y - x) + 0.5 * L * la.norm(y - x, 2) ** 2 + reg_coef * la.norm(y, 1)

    w = w0
    L = L0
    n = y.size

    X_ = 1 / n * X.T.dot(X)
    y_ = 1 / n * X.T.dot(y)

    n_iter = 0

    while True :

        L = L / 2
        w_pred = w
        grad = (X_.dot(w_pred) - y_)
        w = prox(w_pred - 1 / L * grad, 1 / L * reg_coef)

        while func(w) > m_L(w, w_pred, grad, L) :
            L *= 2
            w = prox(w_pred - 1 / L * grad, 1 / L * reg_coef)
        L = max(L0, L)

        mu = min(1, reg_coef / la.norm(X_.dot(w) - y_, np.inf)) * (X.dot(w)  - y) / n
        dual_gap = 0.5 / n * la.norm(X.dot(w) - y) ** 2 + reg_coef * la.norm(w, 1) + 0.5 * n * la.norm(mu) ** 2 + mu.dot(y)

        n_iter += 1
        if dual_gap < tol or n_iter == max_iter :
            break

    status = 0 if dual_gap < tol else 1
    return w, status


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, -1, 1, 0])
reg_coef = 0.01
w0 = np.array([1, 1])
w0_plus = np.array([1, 1])
w0_minus = np.array([2, 2])

w,s = barrier(X, y, reg_coef, w0_plus, w0_minus, max_iter_inner = 6)
print(w)
print(s)

w,s = subgrad(X, y, reg_coef, w0)
print(w)
print(s)

w,s = prox_grad(X, y, reg_coef, w0)
print(w)
print(s)
