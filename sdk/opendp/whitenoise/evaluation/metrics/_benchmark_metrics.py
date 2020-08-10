from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.params._dataset_params import DatasetParams
from opendp.whitenoise.evaluation.params._benchmark_params import BenchmarkParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from opendp.whitenoise.evaluation.metrics._benchmark_metrics import BenchmarkMetrics

class BenchmarkMetrics:
	"""
	Defines the fields available in the metrics payload object
	"""
	def __init__(self, 
			pa : PrivacyAlgorithm,
			algorithm : object,
			privacy_params : PrivacyParams,
			dataset_params : DatasetParams,
			eval_params : EvaluatorParams, 
			metrics : Metrics
		):
		self.pa = pa
		self.algorithm = algorithm
		self.dataset_params = dataset_params
		self.privacy_params = privacy_params
		self.eval_params = eval_params
		self.metrics = metrics