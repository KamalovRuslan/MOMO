import numpy as np


def pack(X_list):
    x = np.array([])
    for X in X_list:
        x = np.append(x, X)
    return x


def unpack(x, arch):
    n_layers = arch['n_layers']
    sizes = arch['sizes']

    X_list = [None] * (n_layers-1)

    i_from = 0
    for k in range(n_layers-1):
        size_cur = sizes[k]
        size_next = sizes[k+1]
        X_list[k] = x[i_from : i_from + size_cur * size_next].reshape((size_next, size_cur))
        i_from += size_cur * size_next

    return X_list


def n_params_total(arch):
    d = sum(arch['sizes'][i] * arch['sizes'][i+1] for i in range(arch['n_layers']-1))
    return d


def compute_vals(A, X_list, arch):
    n_layers = arch['n_layers']
    afuns = arch['afuns']

    Z_list = [None] * n_layers
    U_list = [None] * (n_layers-1)

    Z_list[0] = A.T
    for k in range(n_layers-1):
        U_list[k] = X_list[k].dot(Z_list[k])
        Z_list[k+1] = afuns[k](U_list[k])

    return Z_list, U_list


def compute_grad(X_list, Z_list, U_list, Gz, arch):
    n_layers = arch['n_layers']
    dafuns = arch['dafuns']

    G_list = [None] * (n_layers-1)

    for k in range(n_layers-2, -1, -1):
        Gu = dafuns[k](U_list[k]) * Gz
        Gz = X_list[k].T.dot(Gu)

        G_list[k] = Gu.dot(Z_list[k].T)

    return G_list


def loss_func(x, A, arch):
    X_list = unpack(x, arch)

    # Compute function value
    Z_list, U_list = compute_vals(A, X_list, arch)
    f = (1/(2*A.shape[0])) * np.sum((Z_list[-1] - A.T)**2)

    # Compute function gradient
    Gz = (1/A.shape[0]) * (Z_list[-1] - A.T)
    G_list = compute_grad(X_list, Z_list, U_list, Gz, arch)
    g = pack(G_list)

    # Return result
    return f, g


class LossFuncSum:
    def __init__(self, A, arch):
        self.A = A
        self.arch = arch

        self.n_funcs = A.shape[0]

    def call_ith(self, i, x):
        return loss_func(x, self.A[[i], :], self.arch)
