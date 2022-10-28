import math

from opendp.transformations import make_bounded_sum, make_clamp
from .base import AdditiveNoiseMechanism, Mechanism
from opendp.mod import binary_search_param, enable_features
from opendp.measurements import make_base_discrete_gaussian, make_base_gaussian
from opendp.accuracy import gaussian_scale_to_accuracy
from opendp.combinators import make_zCDP_to_approxDP, make_fix_delta
from .normal import _normal_dist_inv_cdf

class DiscreteGaussian(AdditiveNoiseMechanism):
    def __init__(
            self, epsilon, *ignore, delta, sensitivity=None, max_contrib=1, upper=None, lower=None, **kwargs
        ):
        super().__init__(
                epsilon,
                mechanism=Mechanism.discrete_gaussian,
                delta=delta,
                sensitivity=sensitivity,
                max_contrib=max_contrib,
                upper=upper,
                lower=lower
            )
        if delta <= 0.0:
            raise ValueError("Discrete gaussian mechanism delta must be greater than 0.0")
        self._compute_noise_scale()
    def _compute_noise_scale(self):
        if self.scale is not None:
            return
        lower = self.lower
        upper = self.upper
        max_contrib = self.max_contrib
        bounds = (float(math.floor(lower)), float(math.ceil(upper)))

        rough_scale = float(upper - lower) * max_contrib * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
        if rough_scale > 10_000_000:
            raise ValueError(f"Noise scale is too large using epsilon={self.epsilon} and bounds ({lower}, {upper}) with {self.mechanism}.  Try preprocessing to reduce senstivity, or try different privacy parameters.")
        enable_features('floating-point', 'contrib')
        bounded_sum = (
            make_clamp(bounds=bounds) >>
            make_bounded_sum(bounds=bounds)
        )
        try:
            def make_dp_sum(scale):
                adp = make_zCDP_to_approxDP(make_base_gaussian(scale))
                return bounded_sum >> make_fix_delta(adp, delta=self.delta)
            discovered_scale = binary_search_param(
                lambda s: make_dp_sum(scale=s),
                d_in=1,
                d_out=(self.epsilon, self.delta))
        except Exception as e:
            raise ValueError(f"Unable to find appropriate noise scale for with {self.mechanism} with epsilon={self.epsilon} and bounds ({lower}, {upper}).  Try preprocessing to reduce senstivity, or try different privacy parameters.\n{e}")
        self.scale = discovered_scale
    @property
    def threshold(self):
        max_contrib = self.max_contrib
        delta = self.delta
        if delta == 0.0:
            raise ValueError("censor_dims requires delta to be > 0.0  Try delta=1/n*sqrt(n) where n is the number of individuals")
        thresh = 1 + self.scale * _normal_dist_inv_cdf((1 - delta / 2) ** (1 / max_contrib))
        return thresh
    def release(self, vals):
        enable_features('contrib')
        meas = make_base_discrete_gaussian(self.scale)
        vals = [meas(int(round(v))) for v in vals]
        return vals
    def accuracy(self, alpha):
        return gaussian_scale_to_accuracy(self.scale, alpha)
        
