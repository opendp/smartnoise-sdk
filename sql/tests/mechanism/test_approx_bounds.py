from snsql.sql._mechanisms.approx_bounds import approx_bounds
import numpy as np

class TestApproximateBounds:
    def test_bounds_small(self):
        vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 50
        min, max = approx_bounds(vals, 1.0)
        assert(min < 2)
        assert(max > 7)
    def test_bounds_large_positive(self):
        vals = [2**10, 2**20, 2**30, 2**40] * 100
        min, max = approx_bounds(vals, 1.0)
        assert(min > 2**8 and min < 2**12)
        assert(max > 2**38 and max < 2**42)
    def test_bounds_large_negative(self):
        vals = [-1*2**10, -1*2**20, -1*2**30, -1*2**40] * 100
        min, max = approx_bounds(vals, 1.0)
        assert(max > -1 * 2**12 and max < -1 * 2**8)
        assert(min < -1 * 2**38 and min > -1 * 2**42)
    def test_bounds_large_positive_negative(self):
        vals = [2**10, -1 * 2**20, -1 * 2**30, 2**40, -0.5, -0.75] * 100
        min, max = approx_bounds(vals, 1.0)
        assert(min > -1 * 2**32 and min < -1 * 2**28)
        assert(max > 2**38 and max < 2**42)
    def test_bounds_zero(self):
        vals = np.arange(1000) / 2000
        min, max = approx_bounds(vals, 0.1)
        assert (min == 0.0)
        assert (max == 1.0)
    def test_bounds_zero_negative(self):
        vals = np.arange(1000) / 2000 * -1
        min, max = approx_bounds(vals, 0.1)
        assert (min == -1.0)
        assert (max == 0.0)
    def test_bounds_increment(self):
        powers = np.arange(10) * 4.0
        vals = [2.0**p for p in powers] * 100
        min, max = approx_bounds(vals, 10.0)
        assert (min == 1.0)
        assert (max >= 2**35 and max <= 2**37)
    def test_bounds_ragged_edges(self):
        vals = [1, 3, 5, -6, -4.5, 25, -3, 299, 899] * 100
        min, max = approx_bounds(vals, 1.0)
        assert (min < -6 and max > 899)
    def test_bounds_extreme_values(self):
        vals = [-1 * 2 ** 65, 2.0**65] * 50
        min, max = approx_bounds(vals, 1.0)
        assert min < 10E-17
        assert max > 10E17

        vals = [-1 * 2 ** 65, 2.0**65, -np.inf, np.inf, np.nan] * 50
        min, max = approx_bounds(vals, 1.0)
        assert min < 10E-17
        assert max > 10E17
    def test_bounds_nan(self):
        vals = [-np.inf, np.inf, np.nan] * 50
        min, max = approx_bounds(vals, 1.0)
        assert min is None
        assert max is None
