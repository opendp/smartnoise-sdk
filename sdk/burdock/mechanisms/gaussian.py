import math

from .rand import normal as rand_normal
from .base import AdditiveNoiseMechanism
from scipy.stats import norm


class Gaussian(AdditiveNoiseMechanism):
    def __init__(self, eps, delta=1.0E-16, sensitivity=1.0, tau=1, rows=None):
        super().__init__(eps, delta, sensitivity, tau, rows)
        self.sd = (math.sqrt(math.log(1/delta)) + math.sqrt(math.log(1/delta) + self.eps)) / (math.sqrt(2) * self.eps)

    def release(self, vals):
        noise = rand_normal(0.0, self.sd * self.tau * self.sensitivity, len(vals))
        return noise + vals

    def bounds(self, pct=0.95, bootstrap=False):
        if not bootstrap:
            edge = (1 - pct) / 2.0
            return norm.ppf([edge, 1 - edge], 0.0, self.sd * self.tau * self.sensitivity)
        else:
            return super().bounds(pct, bootstrap)

