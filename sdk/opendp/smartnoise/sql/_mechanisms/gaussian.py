import math

from .rand import normal as rand_normal
from .base import AdditiveNoiseMechanism
from scipy.stats import norm
from opendp.smartnoise.report import Result, Interval, Intervals
from opendp.smartnoise.core import core_library

class Gaussian(AdditiveNoiseMechanism):
    def __init__(self, epsilon, delta=1.0E-16, sensitivity=1.0, max_contrib=1):
        super().__init__(epsilon, delta, sensitivity, max_contrib)
        self.sd = float(sensitivity) * max_contrib * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    def release(self, vals, compute_accuracy=False, bootstrap=False):
        reported_vals = [core_library.gaussian_mechanism(float(v), float(self.eps), float(self.delta), float(self.sensitivity) * self.max_contrib, False) for v in vals]
        mechanism = "Gaussian"
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
        return Result(mechanism, statistic, source, reported_vals, epsilon, delta, sensitivity, self.sd, max_contrib, intervals)
