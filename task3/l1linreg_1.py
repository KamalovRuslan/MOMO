import numpy as np
import scipy as sp
from numpy import linalg as la
import time
from sklearn.datasets import load_svmlight_file

def barrier(X, y, reg_coef, w0_plus, w0_minus, tol=1e-5, tol_inner=1e-7, max_iter=100, max_iter_inner=20, t0=1, gamma=10, c1=1e-4, disp=False, trace=False) :

    n = y.size
    A = X.T.dot(X) / n
    d = X.shape[1]
    y_ = X.T.dot(y) / n
    tau = t0

    def matr(A,w_plus,w_minus, tau) :
        d = w_plus.size
        res = np.array([A[:,i] * (1 + (w_minus[i] / w_plus[i]) ** 2) for i in range(d)]).T
        for i in range(d) :
            res[i][i] += (1 / (tau * w_plus[i] ** 2))
        return res

    def generate_b(w_plus, w_minus, tau):
        b = w_plus - w_minus + 2 * tau *  reg_coef * np.array([w ** 2 for w in w_minus])
        b = b - np.array([w_m ** 2 / w_p for w_m, w_p in zip(w_minus, w_plus)]) - w_minus
        b = -A.dot(b) + y_ + np.array([1 / (tau * w) - reg_coef for w in w_plus])
        return b

    def select_alpha(w,p) :
        alpha_list = [-w[i] / p[i] for i in range(len(w)) if p[i] < 0]
        alpha = np.inf if not alpha_list else min(alpha_list)
        return alpha

    w_plus = w0_plus
    w_minus = w0_minus

    n_iter = 0
    status = 1

    t_start = time.time()
    elaps_t_list = list()
    phi_list = list()
    dual_gap_list = list()

    mu = min(1, n * reg_coef / (la.norm(X.dot(w_plus - w_minus) - y, np.inf))) * (X.dot(w_plus - w_minus) - y) / n
    dual_gap = 0.5 / n * la.norm(X.dot(w_plus - w_minus) - y) ** 2 + reg_coef * la.norm(w_plus - w_minus, 1) + n / 2 * la.norm(mu) ** 2 + mu.dot(y)

    while True :

        grad_plus = A.dot(w_plus - w_minus) - y_
        grad_plus = grad_plus + np.array([reg_coef - 1 / (tau * w) for w in w_plus])
        grad_minus = -A.dot(w_plus - w_minus) + y_
        grad_minus = grad_minus + np.array([reg_coef - 1 / (tau * w) for w in w_minus])
        norm_g = la.norm(np.array(list(grad_plus) + list(grad_minus)))
        n_iter_inner = 0
        if disp:
            print('%s %3d' % ('#:', n_iter))

        while norm_g / tol_inner > 1 / tau and n_iter_inner < max_iter_inner :

            b = generate_b(w_plus, w_minus, tau)
            p_plus = la.solve(matr(A,w_plus,w_minus, tau), b)
            p_minus = np.array([w_m ** 2 * (-2 * reg_coef * tau + 1 / w_p) + w_m - (w_m ** 2) / (w_p ** 2) * p for w_m,w_p,p in zip(w_minus, w_plus, p_plus)])
            dot = grad_plus.dot(p_plus) + grad_minus.dot(p_minus)

            alpha = select_alpha(list(w_plus) + list(w_minus), list(p_plus) + list(p_minus))
            alpha = min(1, 0.95 * alpha)

            scal_1 = A.dot(w_plus - w_minus).dot(p_plus - p_minus) - y_.dot(p_plus - p_minus)
            scal_2 = 0.5 * (la.norm(X.dot(p_plus - p_minus)) ** 2) / n
            scal_3 = reg_coef * sum(list(p_plus + p_minus))
            log_sum = sum([np.log(1 + alpha * p / w) for p,w in zip(p_plus, w_plus)])
            log_sum += sum([np.log(1 + alpha * p / w) for p,w in zip(p_minus, w_minus)])
            armiho = alpha * scal_1 + (alpha ** 2) * scal_2 + alpha * scal_3 - log_sum / tau - c1 * alpha * dot < 0

            while not armiho:
                alpha /= 2
                log_sum = sum([np.log(1 + alpha * p / w) for p,w in zip(p_plus, w_plus)])
                log_sum += sum([np.log(1 + alpha * p / w) for p,w in zip(p_minus, w_minus)])
                armiho = alpha * scal_1 + (alpha ** 2) * scal_2 + alpha * scal_3 - log_sum / tau - c1 * alpha * dot < 0

            w_plus = w_plus + alpha * p_plus
            w_minus = w_minus + alpha * p_minus

            grad_plus = A.dot(w_plus - w_minus) - y_
            grad_plus = grad_plus + np.array([reg_coef - 1 / (tau * w) for w in w_plus])
            grad_minus = -A.dot(w_plus - w_minus) + y_
            grad_minus = grad_minus + np.array([reg_coef - 1 / (tau * w) for w in w_minus])
            norm_g = la.norm(np.array(list(grad_plus) + list(grad_minus)))
            f_min = tau * 0.5 / n * la.norm(X.dot(w_plus - w_minus) - y) ** 2 + tau * reg_coef * sum(list(w_plus + w_minus))
            f_min -= sum([np.log(w_p) + np.log(w_m) for w_p,w_m in zip(w_plus, w_minus)])

            n_iter_inner += 1
            elaps_t = time.time() - t_start

            elaps_t_list.append(elaps_t)
            phi_list.append(f_min)
            dual_gap_list.append(dual_gap)

            if disp:
                print('%s %3d %s %8f %s %5f %s %5f' % ('#_inner:', n_iter_inner,'   norm_g:',norm_g,'   f_min:',f_min,'   elaps_t:',elaps_t))

        mu = min(1, n * reg_coef / (la.norm(X.dot(w_plus - w_minus) - y, np.inf))) * (X.dot(w_plus - w_minus) - y) / n
        dual_gap = 0.5 / n * la.norm(X.dot(w_plus - w_minus) - y) ** 2 + reg_coef * la.norm(w_plus - w_minus, 1) + n / 2 * la.norm(mu) ** 2 + mu.dot(y)
        if disp:
            print('%s %10f' % ('dual_gap', dual_gap))


        n_iter += 1

        if dual_gap <= tol or  n_iter == max_iter :
            status = 0 if dual_gap < tol else 1
            break

        tau *= gamma
    if trace :
        hist = {'elaps_t' : np.array(elaps_t_list), 'phi' : np.array(phi_list), 'dual_gap' : np.array(dual_gap_list)}
        return w_plus - w_minus, status, hist
    else :
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
    real_step = 0
    t_start = time.time()
    elaps_t_list = list()
    phi_list = list()
    dual_gap_list = list()

    while dual_gap > tol and n_iter < max_iter :

        w_pred = w
        f_pred = f

        direction = reg_coef * subg(w) + direction_
        direction /= la.norm(direction)
        w = w - alpha / (real_step + 1) ** 0.5 * direction
        f = func(w)

        if f_pred > f :
            mu = min(1 , reg_coef / la.norm(A.dot(w) - y_), np.inf) * (X.dot(w) - y) / n
            dual_gap = 0.5 / n * la.norm(X.dot(w) - y) ** 2 + reg_coef * la.norm(w, 1) + 0.5 * n * la.norm(mu) ** 2 + mu.dot(y)
            direction_ = A.dot(w) - y_
            real_step += 1
        else :
            w = w_pred
            f = f_pred

        n_iter += 1
        elaps_t = time.time() - t_start

        elaps_t_list.append(elaps_t)
        phi_list.append(f)
        dual_gap_list.append(dual_gap)

        if disp:
            print('%s %3d %s %8f %s %5f %s %5f' % ('#:', n_iter,'   f_min:',f,'   dual_gap:',dual_gap,'   elaps_t:',elaps_t))

    status = 0 if dual_gap < tol else 1
    if trace :
        hist = {'elaps_t' : np.array(elaps_t_list), 'phi' : np.array(phi_list), 'dual_gap' : np.array(dual_gap_list)}
        return w, status, hist
    else :
        return w, status



def prox_grad(X, y, reg_coef, w0, tol=1e-5, max_iter=1000, L0=1, disp=False, trace=False) :

    r = 1 / reg_coef

    def prox(x, alpha) :
        ans = [x[i] + alpha if x[i] < -alpha else x[i] - alpha if x[i] > alpha else 0 for i in range(x.size)]
        return np.array(ans)

    def func(w) :
        return r * 0.5 / n * la.norm(X.dot(w) - y) ** 2 + la.norm(w,1)

    def m_L(y , x, grad, L) :
         return func(x) + r * grad.dot(y - x) + 0.5 * L * la.norm(y - x) ** 2 + la.norm(y, 1)

    w = w0
    L = L0
    n = y.size

    X_ = 1 / n * X.T.dot(X)
    y_ = 1 / n * X.T.dot(y)

    n_iter = 0

    t_start = time.time()
    elaps_t_list = list()
    phi_list = list()
    dual_gap_list = list()
    ls_iters_list = list()
    ls_iters = 0

    while True :

        L = L / 2
        w_pred = w
        grad = (X_.dot(w_pred) - y_)
        w = prox(w_pred - 1 / L * r * grad, 1 / L)
        f = func(w)

        while f > m_L(w, w_pred, grad, L) :
            ls_iters += 1
            L *= 2
            w = prox(w_pred - 1 / L * r * grad, 1 / L)
            f = func(w)
        L = max(L0, L)

        mu = min(1, reg_coef / la.norm(X_.dot(w) - y_, np.inf)) * (X.dot(w)  - y) / n
        dual_gap = 0.5 / n * la.norm(X.dot(w) - y) ** 2 + reg_coef * la.norm(w, 1) + 0.5 * n * la.norm(mu) ** 2 + mu.dot(y)


        n_iter += 1
        elaps_t = time.time() - t_start

        elaps_t_list.append(elaps_t)
        phi_list.append(f / r)
        dual_gap_list.append(dual_gap)
        ls_iters_list.append(ls_iters)

        if disp:
            print('%s %3d %s %8f %s %5f %s %5f' % ('#:', n_iter,'   f_min:',f,'   dual_gap:',dual_gap,'   elaps_t:',elaps_t))

        if dual_gap < tol or n_iter == max_iter :
            break

    status = 0 if dual_gap < tol else 1
    if trace :
        hist = {'elaps_t' : np.array(elaps_t_list), 'phi' : np.array(phi_list), 'dual_gap' : np.array(dual_gap_list), 'ls_iters' : np.array(ls_iters_list)}
        return w, status, hist
    else :
        return w, status

X,y = load_svmlight_file("E:\Programms\Py\MOMO\Data\sonar_scale.svmlight")
X = X.toarray()
y = np.random.uniform(0, 1, (X.shape[0]))
#y = np.zeros(X.shape[0])
w0_plus = 3 * np.ones(60)
w0_minus = 4 * np.ones(60)
w0 = 3 * np.ones(60)
w, s = barrier(X, y, 1 / X.shape[0], w0_plus, w0_minus, tol=1e-9, tol_inner=1e-7, max_iter=100, max_iter_inner=20, t0=6, gamma=10, c1=1e-4, disp=True)
#print(w)
#print(s)
#w,s = subgrad(X, y, 1, w0,disp=True)
#w,s = subgrad(X, y, 1/208, w, disp=True)
print(w)
print(s)
