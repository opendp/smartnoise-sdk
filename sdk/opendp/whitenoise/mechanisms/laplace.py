from .rand import laplace as rand_laplace
from .base import AdditiveNoiseMechanism
from scipy.stats import laplace
from opendp.whitenoise.report import Result, Interval, Intervals


class Laplace(AdditiveNoiseMechanism):
    def __init__(self, eps, delta=None, sensitivity=1.0, max_contrib=1, interval_widths=[0.95], n_rows=None):
        super().__init__(eps, 0, sensitivity, max_contrib, interval_widths, n_rows)
        self.scale = (self.max_contrib * self.sensitivity) / self.eps

    def release(self, vals, compute_accuracy=False, bootstrap=False):
        noise = rand_laplace(0.0, self.scale, len(vals))
        reported_vals = [n + v for n, v in zip(noise, vals)]
        mechanism = "Laplace"
        statistic = "additive_noise"
        source = None
        epsilon = self.eps
        delta = self.delta
        sensitivity = self.sensitivity
        max_contrib = self.max_contrib
        accuracy = None
        intervals = None
        if compute_accuracy:
            bounds = self.bounds(bootstrap)
            accuracy = [(hi - lo) / 2.0 for lo, hi in bounds]
            intervals = Intervals([Interval(confidence, accuracy) for confidence, accuracy in zip(self.interval_widths, accuracy)])
            intervals.extend(reported_vals)
        return Result(mechanism, statistic, source, reported_vals, epsilon, delta, sensitivity, self.scale, max_contrib, intervals)


    def bounds(self, bootstrap=False):
        if not bootstrap:
            _bounds = []
            for a in self.interval_widths:
                edge = (1 - a) / 2.0
                _bounds.append( laplace.ppf([edge, 1 - edge], 0.0, self.scale))
            return _bounds
        else:
            return super().bounds(bootstrap)
