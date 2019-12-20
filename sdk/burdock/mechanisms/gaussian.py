import numpy as np
import math
from scipy.stats import norm


class Gaussian:
    def __init__(self, eps, delta=1.0E-16, tau=1, rows=None):
        self.tau = tau
        self.eps = eps
        self.delta = delta
        if rows is not None:
            delta = 1 / (rows * math.sqrt(rows))
        self.sd = (math.sqrt(math.log(1/delta)) + math.sqrt(math.log(1/delta) + self.eps)) / (math.sqrt(2) * self.eps)

    def count(self, vals):
        noise = np.random.normal(0.0, self.sd * self.tau, len(vals))
        new_vals = noise + vals
        return new_vals

    def sum_int(self, vals, sensitivity):
        noise = np.random.normal(0.0, self.tau * sensitivity * self.sd, len(vals))
        return np.array(noise).astype(int) + vals

    def sum_float(self, vals, sensitivity):
        noise = np.random.normal(0.0, self.tau * sensitivity * self.sd, len(vals))
        return noise + vals
