import numpy as np
import math
from burdock.mechanisms.base import AdditiveNoiseMechanism
from scipy.stats import norm


class Gaussian(AdditiveNoiseMechanism):
    def __init__(self, eps, delta=1.0E-16, sensitivity=1.0, max_contrib=1, alpha=[0.95], rows=None):
        super().__init__(eps, delta, sensitivity, max_contrib, alpha, rows)
        self.sd = (math.sqrt(math.log(1/delta)) + math.sqrt(math.log(1/delta) + self.eps)) / (math.sqrt(2) * self.eps)

    def release(self, vals):
        noise = np.random.normal(0.0, self.sd * self.max_contrib * self.sensitivity, len(vals))
        return noise + vals

    def bounds(self, bootstrap=False):
        if not bootstrap:
            _bounds = []
            for a in self.alpha:
                edge = (1.0 - a) / 2.0
                _bounds.append(norm.ppf([edge, 1 - edge], 0.0, self.sd * self.max_contrib * self.sensitivity))
            return _bounds
        else:
            return super().bounds(bootstrap)

