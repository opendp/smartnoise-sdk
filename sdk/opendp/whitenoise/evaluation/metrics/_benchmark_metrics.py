from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.params._dataset_params import DatasetParams
from opendp.whitenoise.evaluation.params._benchmark_params import BenchmarkParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics

class BenchmarkMetrics:
	"""
	Defines the fields available in the metrics payload object
	"""
	def __init__(self, 
			pa : PrivacyAlgorithm,
			algorithm : object,
			exact_algorithm : object,
			privacy_params : PrivacyParams,
			dataset_params : DatasetParams,
			eval_params : EvaluatorParams, 
			key_metrics : {str : Metrics}
		):
		self.pa = pa
		self.algorithm = algorithm
		self.exact_algorithm = algorithm
		self.privacy_params = privacy_params
		self.dataset_params = dataset_params
		self.eval_params = eval_params
		self.key_metrics = key_metrics