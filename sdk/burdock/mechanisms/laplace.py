import numpy as np
from burdock.mechanisms.base import AdditiveNoiseMechanism
from scipy.stats import laplace
from burdock.metadata.release import Result, Interval


class Laplace(AdditiveNoiseMechanism):
    def __init__(self, eps, delta=None, sensitivity=1.0, max_contrib=1, alphas=[0.95], rows=None):
        super().__init__(eps, 0, sensitivity, max_contrib, alphas, rows)
        self.scale = (self.max_contrib * self.sensitivity) / self.eps

    def release(self, vals, compute_accuracy=False, bootstrap=False):
        noise = np.random.laplace(0.0, self.scale, len(vals))
        reported_vals = noise + vals
        mechanism = "Laplace"
        statistic = "additive_noise"
        source = None
        epsilon = self.eps
        delta = self.delta
        sensitivity = self.sensitivity
        max_contrib = self.max_contrib
        alphas = self.alphas
        accuracy = None
        intervals = None
        if compute_accuracy:
            bounds = self.bounds(bootstrap)
            accuracy = [(hi - lo) / 2.0 for hi, lo in bounds]
            intervals = [[Interval(v - a, v + a) for v in reported_vals] for a in accuracy]
        return Result(mechanism, statistic, source, reported_vals, epsilon, delta, sensitivity, max_contrib, alphas, accuracy, intervals)


    def bounds(self, bootstrap=False):
        if not bootstrap:
            _bounds = []
            for a in self.alphas:
                edge = (1 - a) / 2.0
                _bounds.append( laplace.ppf([edge, 1 - edge], 0.0, self.scale))
            return _bounds
        else:
            return super().bounds(bootstrap)
