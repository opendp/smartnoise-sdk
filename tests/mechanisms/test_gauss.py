import pytest
import numpy as no
from scipy.stats import norm
from burdock.mechanisms.gaussian import Gaussian

#
#   Unit tests
#
class TestGauss:
    def test_simple_norm(self):
        g = Gaussian(0.1) # epsilon of 0.1
        x = range(10000)
        y1 = g.count(x)
        y2 = g.sum_int(x, 100)
        y3 = g.sum_float(x, 100.0)