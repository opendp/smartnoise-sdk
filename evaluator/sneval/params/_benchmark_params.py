from opendp.smartnoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams


class BenchmarkParams:
    """
	Defines the fields used to set benchmarking parameters
    and consumed by the benchmarking API
    Algorithms are the list of DP algorithms that need to be benchmarked
	"""

    def __init__(
        self,
        pa_algorithms: {PrivacyAlgorithm: [object]},
        privacy_params_list: [PrivacyParams],
        d1_d2_list: [[object, object]],
        eval_params: EvaluatorParams,
    ):
        self.pa_algorithms = pa_algorithms
        self.d1_d2_list = d1_d2_list
        self.privacy_params_list = privacy_params_list
        self.eval_params = eval_params
