import numpy as np
from numpy.linalg import norm


class FuncSumWrapper:
    def __init__(self, fsum):
        self.fsum = fsum
        self.n_funcs = fsum.n_funcs
        self.n_evals = 0

    def call_ith(self, i, x):
        self.n_evals += 1
        return self.fsum.call_ith(i, x)


class Logger:
    def __init__(self, fsum_wrp, record_freq=1):
        self.fsum_wrp = fsum_wrp
        self.record_freq = record_freq

        self.n_evals_last = 0
        self.hist = {'epoch': [], 'f': [], 'norm_g': []}

    def record_point(self, x):
        if (self.fsum_wrp.n_evals == 0 or self.fsum_wrp.n_evals > self.n_evals_last + self.record_freq * self.fsum_wrp.n_funcs):
            # Compute full function value and gradient
            f = 0
            g = 0
            for i in range(self.fsum_wrp.n_funcs):
                fi, gi = self.fsum_wrp.fsum.call_ith(i, x)
                f = f + fi
                g = g + gi
            f /= self.fsum_wrp.n_funcs
            g /= self.fsum_wrp.n_funcs

            epoch = self.fsum_wrp.n_evals / self.fsum_wrp.n_funcs
            norm_g = norm(g, np.inf)

            #print('epoch=%.02f, f=%g, norm_g=%g' % (epoch, f, norm_g))

            self.hist['epoch'].append(epoch)
            self.hist['f'].append(f)
            self.hist['norm_g'].append(norm_g)

            self.n_evals_last = self.fsum_wrp.n_evals

    def get_hist(self):
        for key in self.hist:
            self.hist[key] = np.array(self.hist[key])
        return self.hist
