import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import norm
from collections import deque
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal, assert_array_equal
import unittest
from ddt import ddt, data

from io import StringIO
import sys

from lossfuncs import logistic, logistic_hess_vec
from special import grad_finite_diff, hess_vec_finite_diff
from optim import cg, ncg, lbfgs, lbfgs_compute_dir, hfn

############################################################################################################
# Check if it's Python 3
if not sys.version_info > (3, 0):
    print('You should use only Python 3!')
    sys.exit()

############################################################################################################
######################################### Auxiliary functions ##############################################
############################################################################################################

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

############################################################################################################
############################################# TestLogistic #################################################
############################################################################################################

@ddt
class TestLogistic(unittest.TestCase):
    # Simple data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    X_sparse = csr_matrix(X) # the same matrix but in the scipy CSR format
    y = np.array([1, 1, -1, 1])
    reg_coef = 0.5

    testing_Xs = (X, X_sparse) # try both dense and sparse matrices

    @data(*testing_Xs)
    def test_logistic(self, X):
        """Check if the function value and gradient returned correctly."""
        f, g = logistic(np.array([0, 0]), X, self.y, self.reg_coef)

        self.assertTrue(isinstance(g, np.ndarray))

        assert_almost_equal(f, 0.693, decimal=2)
        assert_array_almost_equal(g, [0, -0.25])

    @data(*testing_Xs)
    def test_hess_vec(self, X):
        """Check if the Hessian-vector product returned correctly."""
        hv = logistic_hess_vec(np.array([0, 0]), np.array([1, 1]), X, self.y, self.reg_coef)

        self.assertTrue(isinstance(hv, np.ndarray))

        assert_array_almost_equal(hv, [0.6875, 0.6875])

############################################################################################################
############################################ TestFiniteDiff ################################################
############################################################################################################

class TestFiniteDiff(unittest.TestCase):
    # Define a simple quadratic function
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    phi = (lambda self, x: (1/2)*x.dot(self.A.dot(x)) + self.b.dot(x))

    def test_grad_finite_diff(self):
        """Check the function returns a correct gradient."""
        g = grad_finite_diff(self.phi, np.array([0, 0]))

        self.assertTrue(isinstance(g, np.ndarray))

        assert_array_almost_equal(g, self.b)

    def test_hess_finite_diff(self):
        """Check the function returns a correct Hessian-vector product."""
        hv = hess_vec_finite_diff(self.phi, np.array([0, 0]), np.array([1, 1]))

        self.assertTrue(isinstance(hv, np.ndarray))

        assert_array_almost_equal(hv, np.array([1, 2]))

############################################################################################################
################################################# TestCG ###################################################
############################################################################################################

class TestCG(unittest.TestCase):
    # Define a simple linear system with A = A' > 0
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    x0 = np.array([0, 0])
    matvec = (lambda self, x: self.A.dot(x))

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_sol, status = cg(self.matvec, self.b, self.x0)

        assert_equal(status, 0)
        self.assertTrue(norm(self.A.dot(x_sol) - self.b, np.inf) <= 1e-4)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        cg(self.matvec, self.b, self.x0, tol=1e-6)

    def test_max_iter(self):
        """Check argument `max_iter` is supported and can be set to None."""
        cg(self.matvec, self.b, self.x0, max_iter=None)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            cg(self.matvec, self.b, self.x0, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_sol, status, hist = cg(self.matvec, self.b, self.x0, trace=True)

        self.assertTrue(isinstance(hist['norm_r'], np.ndarray))

############################################################################################################
############################################## TestNCG #####################################################
############################################################################################################

class TestNCG(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    c = 9.5
    x0 = np.array([0, 0])
    func = (lambda self, x: ((1/2)*x.dot(self.A.dot(x)) - self.b.dot(x) + self.c, self.A.dot(x) - self.b))
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    ################################ Tests #######################################
    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, f_min, status = ncg(self.func, self.x0)

        assert_equal(status, 0)
        self.assertTrue(norm(self.A.dot(x_min) - self.b, np.inf) <= 1e-4)
        self.assertTrue(abs(f_min) <= 1e-8)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        ncg(self.func, self.x0, tol=1e-6)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        ncg(self.func, self.x0, max_iter=15)

    def test_c1(self):
        """Check if argument `c1` is supported."""
        ncg(self.func, self.x0, c1=0.2)

    def test_c2(self):
        """Check if argument `c2` is supported."""
        ncg(self.func, self.x0, c2=0.9)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            ncg(self.func, self.x0, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_min, f_min, status, hist = ncg(self.func, self.x0, trace=True)

        self.assertTrue(isinstance(hist['f'], np.ndarray))
        self.assertTrue(isinstance(hist['norm_g'], np.ndarray))
        self.assertTrue(isinstance(hist['n_evals'], np.ndarray))
        self.assertTrue(isinstance(hist['elaps_t'], np.ndarray))

        assert_equal(len(hist['norm_g']), len(hist['f']))
        assert_equal(len(hist['n_evals']), len(hist['f']))
        assert_equal(len(hist['elaps_t']), len(hist['f']))

        # make sure `hist['n_evals']` is a cumulative sum of integers
        assert_equal(np.round(hist['n_evals']), hist['n_evals'])
        self.assertTrue(np.all(hist['n_evals'] >= 0))
        self.assertTrue(np.all(hist['n_evals'][1:] - hist['n_evals'][:-1] > 0))


############################################################################################################
############################################## TestLBFGS ###################################################
############################################################################################################

class TestLBFGS(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    c = 9.5
    x0 = np.array([0, 0])
    func = (lambda self, x: ((1/2)*x.dot(self.A.dot(x)) - self.b.dot(x) + self.c, self.A.dot(x) - self.b))
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    ################################ Tests #######################################
    def test_lbfgs_compute_dir_empty_hist(self):
        """Check if the direction is the negative gradient when the history is empty."""
        # Prepapre data
        sy_hist = deque()
        g = np.array([1, -1])

        # Run test function
        d = lbfgs_compute_dir(sy_hist, g)

        # Check
        self.assertTrue(isinstance(d, np.ndarray))
        assert_array_equal(d, -g)

    def test_lbfgs_compute_dir(self):
        """Use the history of length 1."""
        # Prepare data
        s = np.array([2, -1])
        y = np.array([1, 1])
        sy_hist = deque([(s, y)])
        g = np.array([1, -1])

        # Run test function
        d = lbfgs_compute_dir(sy_hist, g)

        # Check
        assert_array_equal(d, np.array([-11, 8]))

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, f_min, status = lbfgs(self.func, self.x0)

        assert_equal(status, 0)
        self.assertTrue(norm(self.A.dot(x_min) - self.b, np.inf) <= 1e-4)
        self.assertTrue(abs(f_min) <= 1e-8)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        lbfgs(self.func, self.x0, tol=1e-6)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        lbfgs(self.func, self.x0, max_iter=15)

    def test_m(self):
        """Check if argument `m` is supported."""
        lbfgs(self.func, self.x0, m=1)

    def test_c1(self):
        """Check if argument `c1` is supported."""
        lbfgs(self.func, self.x0, c1=0.2)

    def test_c2(self):
        """Check if argument `c2` is supported."""
        lbfgs(self.func, self.x0, c2=0.1)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            lbfgs(self.func, self.x0, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_min, f_min, status, hist = lbfgs(self.func, self.x0, trace=True)

        self.assertTrue(isinstance(hist['f'], np.ndarray))
        self.assertTrue(isinstance(hist['norm_g'], np.ndarray))
        self.assertTrue(isinstance(hist['n_evals'], np.ndarray))
        self.assertTrue(isinstance(hist['elaps_t'], np.ndarray))

        assert_equal(len(hist['norm_g']), len(hist['f']))
        assert_equal(len(hist['n_evals']), len(hist['f']))
        assert_equal(len(hist['elaps_t']), len(hist['f']))

        # make sure `hist['n_evals']` is a cumulative sum of integers
        assert_equal(np.round(hist['n_evals']), hist['n_evals'])
        self.assertTrue(np.all(hist['n_evals'] >= 0))
        self.assertTrue(np.all(hist['n_evals'][1:] - hist['n_evals'][:-1] > 0))

############################################################################################################
############################################### TestHFN ####################################################
############################################################################################################

class TestHFN(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    c = 9.5
    x0 = np.array([0, 0])
    # no need for `extra` for this simple function
    func = (lambda self, x: ((1/2)*x.dot(self.A.dot(x)) - self.b.dot(x) + self.c, self.A.dot(x) - self.b))
    hess_vec = (lambda self, x, v: self.A.dot(v))
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    ################################ Tests #######################################
    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, f_min, status = hfn(self.func, self.x0, self.hess_vec)

        assert_equal(status, 0)
        self.assertTrue(norm(self.A.dot(x_min) - self.b, np.inf) <= 1e-4)
        self.assertTrue(abs(f_min) <= 1e-8)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        hfn(self.func, self.x0, self.hess_vec, tol=1e-6)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        hfn(self.func, self.x0, self.hess_vec, max_iter=15)

    def test_c1(self):
        """Check if argument `c1` is supported."""
        hfn(self.func, self.x0, self.hess_vec, c1=0.2)

    def test_c2(self):
        """Check if argument `c2` is supported."""
        hfn(self.func, self.x0, self.hess_vec, c2=0.1)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            hfn(self.func, self.x0, self.hess_vec, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_min, f_min, status, hist = hfn(self.func, self.x0, self.hess_vec, trace=True)

        self.assertTrue(isinstance(hist['f'], np.ndarray))
        self.assertTrue(isinstance(hist['norm_g'], np.ndarray))
        self.assertTrue(isinstance(hist['n_evals'], np.ndarray))
        self.assertTrue(isinstance(hist['elaps_t'], np.ndarray))

        assert_equal(len(hist['norm_g']), len(hist['f']))
        assert_equal(len(hist['n_evals']), len(hist['f']))
        assert_equal(len(hist['elaps_t']), len(hist['f']))

        # make sure `hist['n_evals']` is a cumulative sum of integers
        assert_equal(np.round(hist['n_evals']), hist['n_evals'])
        self.assertTrue(np.all(hist['n_evals'] >= 0))
        self.assertTrue(np.all(hist['n_evals'][1:] - hist['n_evals'][:-1] > 0))

############################################################################################################
################################################## Main ####################################################
############################################################################################################

if __name__ == '__main__':
    unittest.main()
