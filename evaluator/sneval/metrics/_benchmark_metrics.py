from sneval.privacyalgorithm._base import PrivacyAlgorithm
from sneval.params._privacy_params import PrivacyParams
from sneval.params._eval_params import EvaluatorParams
from sneval.params._dataset_params import DatasetParams
from sneval.params._benchmark_params import BenchmarkParams
from sneval.metrics._metrics import Metrics


class BenchmarkMetrics:
    """
	Defines the fields available in the metrics payload object
	"""

    def __init__(
        self,
        pa: PrivacyAlgorithm,
        algorithm: object,
        privacy_params: PrivacyParams,
        dataset_params: DatasetParams,
        eval_params: EvaluatorParams,
        key_metrics: {str: Metrics},
    ):
        self.pa = pa
        self.algorithm = algorithm
        self.privacy_params = privacy_params
        self.dataset_params = dataset_params
        self.eval_params = eval_params
        self.key_metrics = key_metrics
