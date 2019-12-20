import pytest
import numpy as no
from scipy.stats import norm
from burdock.mechanisms.laplace import Laplace

#
#   Unit tests
#
class TestLaplace:
    def test_simple_lap(self):
        g = Laplace(0.1) # epsilon of 0.1
        x = range(10000)
        y1 = g.count(x)
        y2 = g.sum_int(x, 100)
        y3 = g.sum_float(x, 100.0)