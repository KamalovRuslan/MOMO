from sklearn.datasets import load_digits
from scipy.special import expit
from autoencoder import LossFuncSum, n_params_total
from optim import sgd, svrg
import numpy as np
from autoencoder import compute_vals, unpack
import matplotlib.pyplot as plt
import random
# Define auxuiliary function for plotting results of autoencoder

def plot_result(x, arch, title, digits=[3, 4, 7], colors=['r', 'b', 'g']):
    X_list = unpack(x, arch)
    Z_list = compute_vals(A, X_list, arch)[0]
    Z = Z_list[arch['n_layers']//2] # middle layer
    plt.figure()
    hs = []
    for dig, col in zip(digits, colors):
        mask = (labels == dig)
        h = plt.scatter(Z[0, mask], Z[1, mask], color=col, s=50, alpha=0.5)
        hs.append(h)
    plt.title(title)
    plt.legend(hs, ['Digit %s'%dig for dig in digits], scatterpoints=1)
    plt.show()

# Load data
digits = load_digits()
A = digits['data']
labels = digits['target']

# Normalize values into segment [0, 1]
A = A / np.max(A)
# Define (element-wise) identity function and its derivative
linfun = (lambda x: x)
dlinfun = (lambda x: np.ones_like(x))
# Define (element-wise) sigmoid function and its derivative
sigmfun = expit
dsigmfun = (lambda x: expit(x) * (1 - expit(x)))
# Describe autoencoder architecture
arch = {
'n_layers': 5, # Number of layers: s
'sizes': [64, 30, 2, 30, 64], # Layer sizes: d_1, d_2, d_3
'afuns': [sigmfun, linfun, sigmfun, sigmfun], # Activation functions: sigma_2, sigma_3
'dafuns': [dsigmfun, dlinfun, dsigmfun, dsigmfun], # Derivatives of act. functions: sigma_2’, sigma_3’
}
# Create function object
fsum = LossFuncSum(A, arch)
# Set up starting point
np.random.seed(0)
#x0 = np.random.randn(n_params_total(arch))
x0 = np.array([random.normalvariate(0,1) for i in range(n_params_total(arch))])
# Run SGD
x_sgd, hist_sgd = sgd(fsum, x0, n_iters=10*fsum.n_funcs, step_size=0.1, trace=True)
#plot_result(x0, arch, 'Initial approximation')
plot_result(x_sgd, arch, 'SGD result')

#plt.figure()
#for h in [1e-4, 1e-3, 1e-2, 1e-1, 1] :
#    x_sgd, hist_sgd = sgd(fsum, x0, n_iters=10*fsum.n_funcs, step_size=h, trace=True)
    #plt.plot(hist_sgd['epoch'], hist_sgd['f'], linewidth=1, alpha = 0.8, label='h =' + str(h))
#    plt.semilogy(hist_sgd['epoch'], hist_sgd['norm_g'], linewidth=1, alpha = 0.8, label='h =' + str(h))
#plt.xlabel('Epoch')
#plt.ylabel('Function value')
#plt.ylabel('Gradient norm')
#plt.grid()
#plt.legend()
#plt.show()


#x_sgd, hist_sgd = sgd(fsum, x0, n_iters=10*fsum.n_funcs, step_size=0.1, trace=True)
#x_svrg, hist_svrg = svrg(fsum, x0, n_stages=2, trace=True)

#plt.figure()
#plt.plot(hist_sgd['epoch'], hist_sgd['norm_g'], linewidth=4, label='SGD')
#plt.plot(hist_svrg['epoch'], hist_svrg['norm_g'], linewidth=4, label='SVRG')
#plt.xlabel('Epoch')
#plt.ylabel('Function value')
#plt.ylabel('Gradient norm')
#plt.grid()
#plt.legend()
#plt.show()

#plt.figure()

#for m in range(2, 10) :
#    x_svrg, hist_svrg = svrg(fsum, x0, n_stages=m, trace=True)
    #plt.plot(hist_svrg['epoch'], hist_svrg['f'], linewidth=1, alpha = 0.8,  label='m=' + str(m))
#    plt.semilogy(hist_svrg['epoch'], hist_svrg['norm_g'], linewidth=1, alpha = 0.8, label='m =' + str(m))
#plt.xlabel('Epoch')
#plt.ylabel('Function value')
#plt.ylabel('Gradient norm')
#plt.grid()
#plt.legend()
#plt.show()

#plot_result(x0, arch, 'Initial approximation')
#plot_result(x_sgd, arch, 'SGD result')
