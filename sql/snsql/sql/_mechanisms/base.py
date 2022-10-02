from enum import Enum
import numpy as np

class Mechanism(Enum):
    # gaussian = 1
    laplace = 2
    geometric = 3 # discrete laplace
    # analytic_gaussian = 4
    discrete_gaussian = 5
    discrete_laplace = 6

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
        raise ValueError(f"We do not support threshold censoring of rare dimensions for {self.mechanism}.  If you need thresholding, use laplace or analytic gaussian")
    def release(self, vals):
        """
        Adds noise and releases values.

        Values must be pre-aggregated.
        """
        raise NotImplementedError("Please implement release on the derived class")
    
class Unbounded(AdditiveNoiseMechanism):
    def __init__(
            self, epsilon, *ignore, delta, sensitivity=None, max_contrib=1, upper=None, lower=None, **kwargs
        ):
        super().__init__(
                epsilon,
                mechanism=Mechanism.discrete_laplace,
                delta=delta,
                sensitivity=sensitivity,
                max_contrib=max_contrib,
                upper=upper,
                lower=lower
            )
        self.sensitivity = np.inf
