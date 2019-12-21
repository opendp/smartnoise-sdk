import math
import numpy as np

class AdditiveNoiseMechanism:
    """
    Adds noise to an exact aggregated quantity.
    """
    def __init__(self, eps, delta=0.0, sensitivity=1.0, tau=1, rows=None):
        """
        Initialize an addititive noise mechanism.

        Parameters:
            eps (float): epsilon, the total privacy budget to use per-release
            delta (float): the delta privacy parameter.  Usually much smaller than 1.
            sensitivity (float): the maximum amount that any individual can affect the output.
                For counts, sensitivity is 1.  For sums, sensitivity is the allowed range
            tau (int): the maximum times an individual may appear in a partition.
            rows (int): can be supplied to cause delta to be computed as a heuristic.
        """
        self.eps = eps
        self.delta = delta
        self.sensitivity = sensitivity
        self.tau = tau
        if rows is not None:
            self.delta = 1 / (math.sqrt(rows) * rows)


    def release(self, vals):
        """
        Adds noise and releases values.

        Values must be pre-aggregated.
        """
        raise NotImplementedError("Please implement release on the derived class")

    def bounds(self, pct=0.95, bootstrap=False):
        """
        Returns the error bounds, centered around 0.
        """
        if not bootstrap:
            raise Exception("Analytic bounds is not implemented on this class")
        else:
            vals = np.repeat(0.0, 10000)
            r = self.release(vals)
            edge = (1 - pct) / 2.0
            return np.percentile(r, [edge * 100, 100 - edge * 100])



class Statistic:
    pass