import numpy as np
from sklearn.datasets import load_svmlight_file
from logistic import LossFuncSum
from optim import sgd, svrg
from logistic import predict_labels
import matplotlib.pyplot as plt

def calc_error(x):
    b_hat = predict_labels(A_test, x)
    err = np.mean(b_test != b_hat)
    return err

A_train, b_train = load_svmlight_file('E:\Programms\Py\MOMO\Data\w5a.txt')

fsum = LossFuncSum(A_train, b_train, reg_coef=1/A_train.shape[0])

x0 = np.zeros(A_train.shape[1])

#plt.figure()
#for h in [1e-4, 1e-3, 1e-2, 1e-1, 1] :
#    x_sgd, hist_sgd = sgd(fsum, x0, n_iters=10*fsum.n_funcs, step_size=h, trace=True)
#    #plt.plot(hist_sgd['epoch'], hist_sgd['f'], linewidth=1, alpha = 0.8, label='h =' + str(h))
#    plt.semilogy(hist_sgd['epoch'], hist_sgd['norm_g'], linewidth=1, alpha = 0.8, label='h =' + str(h))

#for m in range(2, 10) :
    #x_svrg, hist_svrg = svrg(fsum, x0, n_stages=m, trace=True)
    #plt.plot(hist_svrg['epoch'], hist_svrg['f'], linewidth=1, alpha = 0.8,  label='m=' + str(m))
    #plt.semilogy(hist_svrg['epoch'], hist_svrg['norm_g'], linewidth=1, alpha = 0.8, label='m =' + str(m))



#plt.xlabel('Epoch')
#plt.ylabel('Function value')
#plt.ylabel('Gradient norm')
#plt.grid()
#plt.legend()
#plt.show()


#x_sgd, hist_sgd = sgd(fsum, x0, n_iters=10*fsum.n_funcs, step_size=0.01, trace=True)
x_svrg, hist_svrg = svrg(fsum, x0, n_stages=2, trace=True)

#A_test, b_test = load_svmlight_file('E:\Programms\Py\MOMO\Data\w5a.t', n_features=A_train.shape[1])

#print('Initial error: %g' % calc_error(x0))
#print('SGD result: %g' % calc_error(x_sgd))
#print('SVRG result: %g' % calc_error(x_svrg))

#plt.figure()
#plt.plot(hist_sgd['epoch'], hist_sgd['f'], linewidth=4, label='SGD')
#plt.plot(hist_svrg['epoch'], hist_svrg['f'], linewidth=4, label='SVRG')
#plt.xlabel('Epoch')
#plt.ylabel('Function value')
#plt.grid()
#plt.legend()
#plt.show()

plt.figure()
#plt.semilogy(hist_sgd['epoch'], hist_sgd['norm_g'], linewidth=4, label='SGD')
plt.semilogy(hist_svrg['epoch'], hist_svrg['norm_g'], linewidth=4, label='SVRG')
plt.xlabel('Epoch')
plt.ylabel('Gradient norm')
plt.grid()
plt.legend()
plt.show()
