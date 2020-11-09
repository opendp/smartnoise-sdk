import math
import numpy as np

class AdditiveNoiseMechanism:
    """
    Adds noise to an exact aggregated quantity.
    """
    def __init__(self, eps, delta=0.0, sensitivity=1.0, max_contrib=1, interval_widths = [0.95], n_rows=None):
        """
        Initialize an addititive noise mechanism.

        Parameters:
            eps (float): epsilon, the total privacy budget to use per-release
            delta (float): the delta privacy parameter.  Usually much smaller than 1.
            sensitivity (float): the maximum amount that any individual can affect the output.
                For counts, sensitivity is 1.  For sums, sensitivity is the allowed range
            max_contrib (int): the maximum times an individual may appear in a partition.
            rows (int): can be supplied to cause delta to be computed as a heuristic.
        """
        self.eps = eps
        self.delta = delta
        self.sensitivity = sensitivity
        self.max_contrib = max_contrib
        self.interval_widths = interval_widths
        if n_rows is not None:
            self.delta = 1 / (math.sqrt(n_rows) * n_rows)


    def release(self, vals, accuracy=False, bootstrap=False):
        """
        Adds noise and releases values.

        Values must be pre-aggregated.
        """
        raise NotImplementedError("Please implement release on the derived class")

    def bounds(self, bootstrap=False):
        """
        Returns the error bounds, centered around 0.
        """
        if not bootstrap:
            raise Exception("Analytic bounds is not implemented on this class")
        else:
            vals = np.repeat(0.0, 10000)
            r = self.release(vals).values
            _bounds = []
            for a in self.interval_widths:
                edge = (1.0 - a) / 2.0
                _bounds.append(np.percentile(r, [edge * 100, 100 - edge * 100]))
            return _bounds


