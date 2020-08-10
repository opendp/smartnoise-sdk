from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.params._benchmark_params import BenchmarkParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from abc import ABC, abstractmethod

class Benchmarking(ABC):
	"""
	Interface for benchmark DP implementations to interface with tests available
	in evaluator. Evaluator tests for various properties of DP implementation
	like privacy, accuracy, utility and bias. Benchmark will run the evaluator
	for multiple parameters like epsilon, dataset size etc. 
	"""
	@abstractmethod
	def benchmark(self, 
		pa : PrivacyAlgorithm,
		algorithm : object,  
		dataset : object, 
		benchmark_params : BenchmarkParams,
		privacy_params : PrivacyParams,
		eval_params : EvaluatorParams) -> {[BenchmarkParams, PrivacyParams, EvaluatorParams] : Metrics}:
		"""
		Benchmarks properties of privacy algorithm DP implementations using metrics
			- Privacy Promise
			- Accuracy Promise
			- Utility Promise
			- Bias Promise
		
		d1 and d2 are neighboring datasets
		algorithm is the DP implementation object
		Returns a metrics object
		"""
		pass