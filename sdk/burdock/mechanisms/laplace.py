import numpy as np
from burdock.mechanisms.base import AdditiveNoiseMechanism
from scipy.stats import laplace


class Laplace(AdditiveNoiseMechanism):
    def __init__(self, eps, sensitivity=1.0, max_contrib=1, alpha=[0.95], rows=None):
        super().__init__(eps, 0, sensitivity, max_contrib, alpha, rows)
        self.scale = (self.max_contrib * self.sensitivity) / self.eps

    def release(self, vals, accuracy=False, bootstrap=False):
        noise = np.random.laplace(0.0, self.scale, len(vals))
        reported_vals = noise + vals
        mechanism = "Laplace"
        statistic = "additive_noise"
        source = None
        epsilon = self.eps
        delta = None
        sensitivity = self.sensitivity
        max_contrib = self.max_contrib
        alpha = self.alpha
        accuracy = None
        interval = None
        if accuracy:
            bounds = self.bounds(bootstrap)
            accuracy = [(hi - lo) / 2.0 for hi, lo in bounds]
            interval = [[v - a, v + a] for a in accuracy for v in reported_vals]

        return Result(mechanism, statistic, source, reported_vals, epsilon, delta, sensitivity, max_contrib, alpha, accuracy, interval)


    def bounds(self, bootstrap=False):
        if not bootstrap:
            _bounds = []
            for a in self.alpha:
                edge = (1 - a) / 2.0
                _bounds.append( laplace.ppf([edge, 1 - edge], 0.0, self.scale))
            return _bounds
        else:
            return super().bounds(bootstrap)
