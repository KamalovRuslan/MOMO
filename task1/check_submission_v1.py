import numpy as np
from numpy.testing import assert_equal, assert_allclose
import unittest
from ddt import ddt, data, unpack

from io import StringIO
import sys

from optim1d import min_golden, min_parabolic, min_brent, min_secant, min_brent_der

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

class MyList(list):
    pass

def annotated(min_method, min_func):
    r = MyList([min_method, min_func])
    setattr(r, '__name__', 'test_%s' % min_method.__name__)
    return r

############################################################################################################
############################################### Main code ##################################################
############################################################################################################

### Define a simple quadratic problem for testing ###
func = (lambda x: 0.5 * (x + 1)**2)
func_der = (lambda x: (0.5 * (x + 1)**2, x + 1))
a, b = -2, 1
# For this function |x_min - x^*| < tol ensures |f(x_min) - f(x^*)| < tol^2

testing_pairs = (
    annotated(min_golden, func),
    annotated(min_parabolic, func),
    annotated(min_brent, func),
    annotated(min_secant, func_der),
    annotated(min_brent_der, func_der),
)

@ddt
class TestOptim1d(unittest.TestCase):
    @data(*testing_pairs)
    @unpack
    def test_default(self, min_method, min_func):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, f_min, status = min_method(min_func, a, b)

        assert_equal(status, 0)
        self.assertTrue(abs(x_min - (-1)) <= 1e-5)
        self.assertTrue(abs(f_min) <= 1e-10)
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    @data(*testing_pairs)
    @unpack
    def test_tol(self, min_method, min_func):
        """Try high accuracy."""
        x_min, f_min, status = min_method(min_func, a, b, tol=1e-8)

        assert_equal(status, 0)
        self.assertTrue(abs(x_min - (-1)) <= 1e-8)
        self.assertTrue(abs(f_min) <= 1e-14)

    @data(*testing_pairs)
    @unpack
    def test_max_iter(self, min_method, min_func):
        """Check if argument `max_iter` is supported."""
        min_method(min_func, a, b, max_iter=15)

    @data(*testing_pairs)
    @unpack
    def test_disp(self, min_method, min_func):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            min_method(min_func, a, b, disp=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `disp` is True.')

    @data(*testing_pairs)
    @unpack
    def test_trace(self, min_method, min_func):
        """Check if the history is returned correctly when `trace` is True."""
        x_min, f_min, status, hist = min_method(min_func, a, b, trace=True)

        self.assertTrue(isinstance(hist['x'], np.ndarray))
        self.assertTrue(isinstance(hist['f'], np.ndarray))
        self.assertTrue(isinstance(hist['n_evals'], np.ndarray))

        assert_equal(len(hist['f']), len(hist['x']))
        assert_equal(len(hist['n_evals']), len(hist['x']))

        # make sure hist['f'] contains the values func(hist['x'])
        true_f = np.array([func(x) for x in hist['x']])
        assert_allclose(hist['f'], true_f)

        # make sure hist['n_evals'] is a cumulative sum
        self.assertTrue(np.all(hist['n_evals'] >= 0))
        self.assertTrue(np.all(hist['n_evals'][1:] - hist['n_evals'][:-1] > 0))

if __name__ == '__main__':
    unittest.main()
