import pytest
import math
import numpy as np
from scipy.stats import norm
from opendp.smartnoise.mechanisms.laplace import Laplace

#
#   Unit tests
#
class TestLaplace:
    def test_simple_lap(self):
        g = Laplace(0.1) # epsilon of 0.1
        x = range(10000)
        y = g.release(x).values
        assert(round(np.sum(x) / 10E+6) == round(np.sum(y) / 10E+6))

    def test_bounds1_lap(self):
        # check that analytic and bootstrap bounds work
        g = Laplace(0.5) # epsilon of 0.5
        lower, upper = g.bounds(False)[0] # analytic bounds
        lower2, upper2 = g.bounds(True)[0]  # bootstrap bounds
        assert(lower < upper)
        assert(lower2 < upper2)

    def test_bounds1b_lap(self):
        # check that analytic and bootstrap bounds work
        g = Laplace(0.05) # epsilon of 0.05, very wide bounds
        lower, upper = g.bounds(False)[0] # analytic bounds
        lower2, upper2 = g.bounds(True)[0]  # bootstrap bounds
        assert(lower < upper)
        assert(lower2 < upper2)

    def test_bounds1c_lap(self):
        # check that analytic and bootstrap bounds work
        # use very small bounds to make sure order doesn't swap
        g = Laplace(1.0, interval_widths=[0.1]) # epsilon of 1.0
        lower, upper = g.bounds(False)[0] # analytic bounds
        lower2, upper2 = g.bounds(True)[0]  # bootstrap bounds
        assert(lower <= upper)
        assert(lower2 <= upper2)
    def test_bounds2_lap(self):
        # check that outer bounds enclose inner bounds
        g = Laplace(4.0, interval_widths=[0.95, 0.97]) # epsilon of 4.0, tighter bounds
        lower1, upper1 = g.bounds(False)[0]
        lower1b, upper1b = g.bounds(True)[0]
        lower2, upper2 = g.bounds(False)[1]
        lower2b, upper2b = g.bounds(True)[1]
        assert(lower2 < lower1)
        assert(upper2 > upper1)
        assert(lower2b < lower1b)
        assert(upper2b > upper1b)
    def test_intervals_lap(self):
        l = Laplace(4.0, interval_widths=[0.95, 0.5])
        vals = [100, 3333, 99999]
        r_n = l.release(vals, False)
        r = l.release(vals, True)
        assert(r_n.accuracy is None)
        assert(r.accuracy is not None)
        r0, r1 = r.intervals
        assert(r0.contains(r1))
        assert(not r0.inside(r1))
