import numpy as np
from burdock.mechanisms.base import AdditiveNoiseMechanism
from scipy.stats import laplace


class Laplace(AdditiveNoiseMechanism):
    def __init__(self, eps, sensitivity=1.0, tau=1):
        super().__init__(eps, 0, sensitivity, tau, rows=None)
        self.scale = (self.tau * self.sensitivity) / self.eps

    def release(self, vals):
        noise = np.random.laplace(0.0, self.scale, len(vals))
        return noise + vals

    def bounds(self, pct=0.95, bootstrap=False):
        if not bootstrap:
            edge = (1 - pct) / 2.0
            return laplace.ppf([edge, 1 - edge], 0.0, self.scale)
        else:
            return super().bounds(pct, bootstrap)

    def count(self, vals):
        return self.release(vals)

    def sum_int(self, vals, sensitivity):
        noise = np.random.laplace(0.0, (self.tau * sensitivity) / self.eps, len(vals))
        return np.array(noise).astype(int) + vals

    def sum_float(self, vals, sensitivity):
        noise = np.random.laplace(0.0, (self.tau * sensitivity) / self.eps, len(vals))
        return noise + vals
