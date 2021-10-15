from enum import Enum

class Mechanism(Enum):
    gaussian = 1
    laplace = 2
    geometric = 3

class AdditiveNoiseMechanism:
    """
    Adds noise to an exact aggregated quantity.
    """

    def __init__(
        self, epsilon, *ignore, delta=0.0, sensitivity=None, max_contrib=1, upper=None, lower=None, mechanism, **kwargs
    ):
        """
        Initialize an additive noise mechanism.

        Parameters:
            epsilon (float): epsilon, the total privacy budget to use per-release
            delta (float): the delta privacy parameter.  Usually much smaller than 1.
            sensitivity (float): the maximum amount that any individual can affect the output.
                For counts, sensitivity is 1.  For sums, sensitivity is the allowed range
            max_contrib (int): the maximum times an individual may appear in a partition.
            rows (int): can be supplied to cause delta to be computed as a heuristic.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.max_contrib = max_contrib
        self.upper = upper
        self.lower = lower
        self.mechanism = mechanism
        self.scale = None
        if (upper is None or lower is None) and sensitivity is None:
            raise ValueError("Please pass upper and lower bounds, or pass sensitivity")
        if (upper is not None or lower is not None) and sensitivity is not None:
            raise ValueError("Please pass only sensitivity or bounds, not both")
        if sensitivity is not None:
            # better to just pass in bounds
            self.lower = 0
            self.upper = sensitivity
    def _compute_noise_scale(self):
        raise NotImplementedError("Implement _compute_noise_scale in inherited class")
    @property
    def threshold(self):
        raise NotImplementedError(f"Threshold not implemented for {self.mechanism}")
    def release(self, vals):
        """
        Adds noise and releases values.

        Values must be pre-aggregated.
        """
        raise NotImplementedError("Please implement release on the derived class")
    