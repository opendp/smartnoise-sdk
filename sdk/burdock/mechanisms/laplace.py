from .rand import laplace as rand_laplace
from .base import AdditiveNoiseMechanism
from scipy.stats import laplace


class Laplace(AdditiveNoiseMechanism):
    def __init__(self, eps, sensitivity=1.0, tau=1):
        super().__init__(eps, 0, sensitivity, tau, rows=None)
        self.scale = (self.tau * self.sensitivity) / self.eps

    def release(self, vals):
        noise = rand_laplace(0.0, self.scale, len(vals))
        return noise + vals

    def bounds(self, pct=0.95, bootstrap=False):
        if not bootstrap:
            edge = (1 - pct) / 2.0
            return laplace.ppf([edge, 1 - edge], 0.0, self.scale)
        else:
            return super().bounds(pct, bootstrap)


