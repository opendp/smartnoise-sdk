import math

from opendp.trans import make_bounded_sum, make_clamp
from .base import AdditiveNoiseMechanism, Mechanism
from opendp.mod import binary_search_param, enable_features
from opendp.meas import make_base_geometric

class Geometric(AdditiveNoiseMechanism):
    def __init__(
            self, epsilon, *ignore, delta=0.0, sensitivity=None, max_contrib=1, upper=None, lower=None, **kwargs
        ):
        super().__init__(
                epsilon,
                mechanism=Mechanism.geometric,
                delta=0.0,
                sensitivity=sensitivity,
                max_contrib=max_contrib,
                upper=upper,
                lower=lower
            )
        self._compute_noise_scale()
    def _compute_noise_scale(self):
        if self.scale is not None:
            return
        lower = self.lower
        upper = self.upper
        max_contrib = self.max_contrib
        # should probably just check and throw if not int
        bounds = (int(lower), int(upper))

        rough_scale = (float(upper - lower) * max_contrib) / self.epsilon
        if rough_scale > 10_000_000:
            raise ValueError(f"Noise scale is too large using epsilon={self.epsilon} and bounds ({lower}, {upper}) with {self.mechanism}.  Try preprocessing to reduce senstivity, or try different privacy parameters.")
        search_upper = rough_scale * 10E+6
        search_lower = rough_scale / 10E+6

        enable_features('floating-point', 'contrib')
        bounded_sum = (
            make_clamp(bounds=bounds) >>
            make_bounded_sum(bounds=bounds)
        )
        try:
            discovered_scale = binary_search_param(
                lambda s: bounded_sum >> make_base_geometric(scale=s),
                bounds=(search_lower, search_upper),
                d_in=max_contrib,
                d_out=(self.epsilon))
        except Exception as e:
            raise ValueError(f"Unable to find appropriate noise scale for {self.mechanism} with epsilon={self.epsilon} and bounds ({lower}, {upper}).  Try preprocessing to reduce senstivity, or try different privacy parameters.\n{e}")

        self.scale = discovered_scale
    def release(self, vals):
        enable_features('floating-point', 'contrib')
        meas = make_base_geometric(self.scale)
        vals = [meas(int(v)) for v in vals]
        return vals
    def accuracy(self, alpha):
        percentile = 1 - alpha
        prob = math.exp(-1/self.scale)
        pct = (1 - prob)/(1 + prob)
        n = 0
        while pct < percentile:
            n += 1
            pct += 2 * (1 - prob)/(1 + prob) * (prob ** abs(n))
        return n
