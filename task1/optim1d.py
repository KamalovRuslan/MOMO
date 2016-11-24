import math
import numpy as np

def min_golden(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False):
    K = (math.sqrt(5) - 1) / 2
    current_length = (b - a) * K
    x_left = a
    x_right = b
    x_a = x_right - current_length
    x_b = x_left + current_length
    current_iter = 1
    f_a = func(x_a)
    f_b = func(x_b)
    if (disp):
        print('%s %3d %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   f:',min(f_a,f_b),'   x_a:',x_a,'   x_b:',x_b,'   Current length:',current_length))
    x = list()
    f = list()
    n_evals = list()
    n = 4
    while (current_iter < max_iter) and (current_length >= tol):
        current_length *= K
        if (f_a >= f_b):
            x_left = x_a
            x_a = x_b
            x_b = x_left + current_length
            f_a = f_b
            f_b = func(x_b)
            x.append(x_b)
            f.append(f_b)
            n += 1
            n_evals.append(n)
        else :
            x_right = x_b
            x_b = x_a
            x_a = x_right - current_length
            f_b = f_a
            f_a = func(x_a)
            x.append(x_a)
            f.append(f_a)
            n += 1
            n_evals.append(n)
        current_iter += 1
        if (disp):
            print('%s %3d %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   f:',min(f_a,f_b),'   x_a:',x_a,'   x_b:',x_b,'   Current length:',current_length))
    x_min = x_a if (f_a <= f_b) else x_b
    f_min = f_a if (f_a <= f_b) else f_b
    status = 0 if (current_length < tol) else 1
    if (trace):
        hist = {'x': np.array(x), 'f' : np.array(f), 'n_evals' : np.array(n_evals)}
        return x_min, f_min, status, hist
    else :
        return   x_min, f_min, status


def min_parabolic(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False):
    x_1 = a
    x_3 = b
    x_2 = (x_1 + x_3) / 2
    f_1 = func(x_1)
    f_2 = func(x_2)
    f_3 = func(x_3)
    delta = b - a
    current_iter = 1
    x = list()
    f = list()
    n_evals = list()
    n = 3
    if (disp):
        print('%s %3d %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   f:',f_2,'   x_1:',x_1,'   x_2:',x_2,'   x_3:',x_3,'   Current delta:',delta))
    while (current_iter < max_iter) and (delta >= tol):
        x.append(x_2)
        f.append(f_2)
        A = [[x_1 * x_1, x_1, 1],[x_2 * x_2, x_2, 1],[x_3 * x_3, x_3, 1]]
        b = [f_1, f_2, f_3]
        sol = np.linalg.solve(A,b)
        u = -0.5 * sol[1] / sol[0]
        if u >= x_3 :
            u = x_3 - tol
        if u <= x_1 :
            u = x_1 + tol
        delta = abs(x_2 - u)
        if (x_2 >= u):
            x_3 = x_2
            f_3 = f_2
        else :
            x_1 = x_2
            f_1 = f_2
        x_2 = u
        f_2 = func(u)
        n += 1
        n_evals.append(n)
        current_iter += 1
        if (disp):
            print('%s %3d %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   f:',f_2,'   x_1:',x_1,'   x_2:',x_2,'   x_3:',x_3,'   Current delta:',delta))
    status = 0 if (delta < tol) else 1
    x_min = x_2
    f_min = f_2
    if (trace) :
        hist = {'x': np.array(x), 'f' : np.array(f), 'n_evals' : np.array(n_evals)}
        return x_min, f_min, status, hist
    else :
        return  x_min, f_min, status

def min_brent(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False):
    K = (math.sqrt(5) - 1) / 2
    x_left = a
    x_right = b
    x = (0.5 * a + 0.5 * b)
    w = x
    v = x
    f_x = func(x)
    f_v = f_x
    f_w = f_x
    current_step = b - a
    pre_step = current_step
    current_iter = 1
    n = 1
    current_x = list()
    current_f = list()
    n_evals = list()
    if (disp):
        print('%s %3d %s %s %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   method:','bisection','   x:',x,'   w:',w,'   v:',v,'   f_x:',f_x,'   f_w:',f_w,'   f_v:',f_v,'   Current step:',current_step))
    while (current_iter < max_iter) and (current_step > tol):
        current_x.append(x)
        current_f.append(f_x)
        pre_pre_step = pre_step
        pre_step = current_step
        parabolic = False
        accept_u = False
        if ((x != w) and (x != v) and (w != v) and (f_x != f_w) and (f_x != f_v) and (f_w != f_v)):
            A = [[w*w, w, 1],[x*x, x, 1],[v*v, v, 1]]
            b = [f_w,f_x,f_v]
            sol = np.linalg.solve(A,b)
            u = -0.5 * sol[1] / sol[0]
            if (x_left + tol <= u) and (u <= x_right - tol) and (abs(u - x) < 0.5 * pre_pre_step):
                current_step = abs(u-x)
                accept_u = True
                parabolic = True
        if not accept_u :
            if (x < 0.5 * abs(x_right + x_left)):
                u = x + K * (x_right - x)
                current_step = x_right - x
            else :
                u = x - K * (x - x_left)
                current_step = x - x_left
        if (abs(u - x) < tol):
            u = x - tol if (u < x) else x + tol
        f_u = func(u)
        n += 1
        n_evals.append(n)
        if (f_u <= f_x):
            if (u >= x):
                x_left = x
            else :
                x_right = x
            v, w, x, f_v, f_w, f_x = w, x, u, f_w, f_x, f_u
        else :
            if (u >= x):
                x_right = u
            else :
                x_left = u
            if (f_u <= f_w) or (w == x):
                v = w
                w = u
                f_v = f_w
                f_w = f_u
            else :
                if (f_u <= f_v) or (v == x) or (v == w):
                    v = u
                    f_v = f_u
        current_iter += 1
        if (disp):
            if parabolic :
                print('%s %3d %s %s %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   method:','parabolic','   x:',x,'   w:',w,'   v:',v,'   f_x:',f_x,'   f_w:',f_w,'   f_v:',f_v,'   Current step:',current_step))
            else :
                print('%s %3d %s %s %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   method:','golden   ','   x:',x,'   w:',w,'   v:',v,'   f_x:',f_x,'   f_w:',f_w,'   f_v:',f_v,'   Current step:',current_step))
    status = 0 if (current_step < tol) else 1
    f_min = f_x
    x_min = x
    x = current_x
    f = current_f
    if (trace) :
        hist = {'x': np.array(x), 'f' : np.array(f), 'n_evals' : np.array(n_evals)}
        return x_min, f_min, status, hist
    else :
        return  x_min, f_min, status

def min_secant(func, a, b, tol=1e-5, max_iter=500, disp=False, trace=False):
    current_iter = 1
    delta = b - a
    x_left = a
    x_right = b
    f_l, df_l = func(x_left)
    f_r, df_r = func(x_right)
    x_min = (x_left * df_r - x_right * df_l) / (df_r - df_l)
    f_min, df_min = func(x_min)
    x = list()
    f = list()
    n_evals = list()
    n = 3
    step = x_right - x_left
    if (disp):
        print('%s %3d %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   x_left:',x_left,'   x:',x_min,'   x_right:',x_right,'   df_x:',df_min,'   Current step:',step))
    while (current_iter < max_iter) and (abs(df_min) > tol):
        x.append(x_min)
        f.append(f_min)
        if (df_l * df_min >= 0):
            x_left = x_min
            f_l, df_l = f_min, df_min
        else :
            x_right = x_min
            f_r, df_r = f_min, df_min
        x_pred = x_min
        x_min = (x_left * df_r - x_right * df_l) / (df_r - df_l)
        step = abs(x_min - x_pred)
        f_min, df_min = func(x_min)
        n += 1
        n_evals.append(n)
        current_iter += 1
        if (disp):
            print('%s %3d %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   x_left:',x_left,'   x:',x_min,'   x_right:',x_right,'   df_x:',df_min,'   Current step:',step))

    status = 0 if (df_min < tol) else 1
    if (trace) :
        hist = {'x': np.array(x), 'f' : np.array(f), 'n_evals' : np.array(n_evals)}
        return x_min, f_min, status, hist
    else :
        return  x_min, f_min, status

def min_brent_der(func, a, b, tol=1e-05, max_iter=500, disp=False, trace=False):
    x = list()
    f = list()
    n_evals = list()
    x_left = a
    x_right = b
    x_min = (x_left + x_right) / 2
    w = x_min
    v = w
    f_min, df_min = func(x_min)
    f_w, df_w = f_min, df_min
    f_v, df_v = f_min, df_min
    current_step = x_right - x_left
    pre_step = current_step
    n = 1
    current_iter = 1
    if (disp):
        print('%s %3d %s %s %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   method:','   bisection','   x:',x_min,'   w:',x_min,'   v:',x_min,'   f_x:',df_min,'   f_w:',df_w,'   f_v:',df_v,'   Current step:',current_step))
    while (current_iter < max_iter) and (current_step > 1.000001 *  tol) :
        x.append(x_min)
        f.append(f_min)
        pre_pre_step = pre_step
        pre_step = current_step
        u1_flag = False
        u2_flag = False
        parabolic = False
        if (x_min != w) and (df_min != df_w):
            A = [[2 * x_min,1,0],[2 * w,1,0],[x_min * x_min, x_min,1]]
            b = [df_min,df_w,f_min]
            sol = np.linalg.solve(A,b)
            u1 = -0.5 * sol[1] / sol[0]
            u1_step = abs(u1 - x_min)
            if (x_left + tol <= u1) and (u1 <= x_right - tol) and (u1_step < 0.5 * pre_pre_step):
                u1_flag = True
        if (x_min != v) and (df_min != df_v):
            A = [[2 * x_min,1,0],[2 * v,1,0],[x_min * x_min, x_min,1]]
            b = [df_min,df_v,f_min]
            sol = np.linalg.solve(A,b)
            u2 = -0.5 * sol[1] / sol[0]
            u2_step = abs(u2 - x_min)
            if (x_left + tol <= u1) and (u1 <= x_right - tol) and (u2_step < 0.5 * pre_pre_step):
                u2_flag = True
        if u1_flag or u2_flag :
            parabolic = True
            if u1_flag and u2_flag:
                u = u1 if (u1_step <= u2_step) else u2
            else :
                u = u1 if u1_flag else u2
        else :
            if df_min > 0 :
                u = (x_left + x_min) / 2
            else :
                u = (x_right + x_min) / 2
        if abs(u - x_min) < tol:
            u = x_min + tol if (u >= x_min) else x_min - tol
        current_step = abs(u - x_min)
        f_u, df_u = func(u)
        n += 1
        n_evals.append(n)
        if f_u <= f_min :
            if u >= x_min:
                x_left = x_min
            else :
                x_right = x_min
            v , w , x_min , f_v, f_w, f_min, df_v, df_w, df_min = w, x_min, u, f_w, f_min, f_u, df_w, df_min, df_u

        else :
            if u >= x_min:
                x_right = u
            else :
                x_left = u
            if f_u <= f_w or w == x_min :
                v , w, f_v, f_w, df_v, df_w = w, u, f_w, f_u, df_w, df_u
            else :
                if f_u <= f_v or v == x_min or w == v :
                      v, f_v, f_w = u, f_u, df_u
        current_iter += 1
        if (disp):
            if parabolic:
                print('%s %3d %s %s %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   method:','   parabolic','   x:',x_min,'   w:',w,'   v:',v,'   f_x:',df_min,'   f_w:',df_w,'   f_v:',df_v,'   Current step:',current_step))
            else:
                print('%s %3d %s %s %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f %s %5f' % ('#:', current_iter,'   method:','   bisection','   x:',x_min,'   w:',w,'   v:',v,'   f_x:',df_min,'   f_w:',df_w,'   f_v:',df_v,'   Current step:',current_step))
    status = 0 if (current_step < 1.000001 * tol) else 1
    if (trace):
        hist = {'x': np.array(x), 'f' : np.array(f), 'n_evals' : np.array(n_evals)}
        return  x_min, f_min, status, hist
    else :
        return x_min, f_min, status
