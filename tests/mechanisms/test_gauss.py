import pytest
import numpy as np
from scipy.stats import norm
from burdock.mechanisms.gaussian import Gaussian

#   Unit tests
#
class TestGauss:
    def test_simple_norm(self):
        g = Gaussian(0.1) # epsilon of 0.1        
        x = range(10000)
        y = g.count(x)
        assert(round(np.sum(x) / 10E+6) == round(np.sum(y) / 10E+6))

    def test_bounds1_norm(self):
        # check that analytic and bootstrap bounds work
        g = Gaussian(0.5, 1/125.0) # epsilon of 0.5
        lower, upper = g.bounds(0.95, False) # analytic bounds
        lower2, upper2 = g.bounds(0.95, True)  # bootstrap bounds
        assert(lower < upper)
        assert(lower2 < upper2)

    def test_bounds1b_norm(self):
        # check that analytic and bootstrap bounds work with tiny epsilon
        g = Gaussian(0.05, (1/125.0)) # epsilon of 0.05, very wide bounds
        lower, upper = g.bounds(0.95, False) # analytic bounds
        lower2, upper2 = g.bounds(0.95, True)  # bootstrap bounds
        assert(lower < upper)
        assert(lower2 < upper2)

    def test_bounds1c_norm(self):
        # check that analytic and bootstrap bounds work
        # use very small bounds to make sure order doesn't swap
        g = Gaussian(1.0) # epsilon of 1.0
        lower, upper = g.bounds(0.1, False) # analytic bounds
        lower2, upper2 = g.bounds(0.1, True)  # bootstrap bounds
        assert(lower <= upper)
        assert(lower2 <= upper2)

    def test_bounds2_norm(self):
        # check that outer bounds enclose inner bounds
        g = Gaussian(4.0) # epsilon of 4.0, tighter bounds
        lower1, upper1 = g.bounds(0.95, False)
        lower1b, upper1b = g.bounds(0.95, True) 
        lower2, upper2 = g.bounds(0.97, False)
        lower2b, upper2b = g.bounds(0.97, True)
        assert(lower2 < lower1)
        assert(upper2 > upper1)
        assert(lower2b < lower1b)
        assert(upper2b > upper1b)
