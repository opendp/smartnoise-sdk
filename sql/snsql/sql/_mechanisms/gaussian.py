import math

from opendp.trans import make_bounded_sum, make_clamp
from .base import AdditiveNoiseMechanism, Mechanism
from opendp.mod import binary_search_param, enable_features
from opendp.meas import make_base_gaussian
from opendp.accuracy import gaussian_scale_to_accuracy

class Gaussian(AdditiveNoiseMechanism):
    def __init__(
            self, epsilon, *ignore, delta, sensitivity=None, max_contrib=1, upper=None, lower=None, **kwargs
        ):
        super().__init__(
                epsilon,
                mechanism=Mechanism.gaussian,
                delta=delta,
                sensitivity=sensitivity,
                max_contrib=max_contrib,
                upper=upper,
                lower=lower
            )
        if delta <= 0.0:
            raise ValueError("Gaussian mechanism delta must be greater than 0.0")
        self._compute_noise_scale()
    def _compute_noise_scale(self):
        if self.scale is not None:
            return
        lower = self.lower
        upper = self.upper
        max_contrib = self.max_contrib
        bounds = (float(lower), float(upper))

        rough_scale = float(upper - lower) * max_contrib * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
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
                lambda s: bounded_sum >> make_base_gaussian(scale=s),
                bounds=(search_lower, search_upper),
                d_in=max_contrib,
                d_out=(self.epsilon, self.delta))
        except Exception as e:
            raise ValueError(f"Unable to find appropriate noise scale for {self.mechanism} with epsilon={self.epsilon} and bounds ({lower}, {upper}).  Try preprocessing to reduce senstivity, or try different privacy parameters.\n{e}")
        self.scale = discovered_scale
    def release(self, vals):
        enable_features('floating-point', 'contrib')
        meas = make_base_gaussian(self.scale)
        vals = [meas(float(v)) for v in vals]
        return vals
    def accuracy(self, alpha):
        return gaussian_scale_to_accuracy(self.scale, alpha)
        
