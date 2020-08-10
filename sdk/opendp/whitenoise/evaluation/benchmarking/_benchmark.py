from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.params._benchmark_params import BenchmarkParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from opendp.whitenoise.evaluation.metrics._benchmark_metrics import BenchmarkMetrics
from opendp.whitenoise.evaluation.benchmarking._base import Benchmarking
from opendp.whitenoise.evaluation.eveluator._dp_eveluator import DPEvaluator

class DPBenchmarking(Benchmarking):
	"""
	Implement interface to benchmark DP implementations to interface with tests available
	in DP evaluator. Evaluator tests for various properties of DP implementation
	like privacy, accuracy, utility and bias. Benchmark will run the evaluator
	for multiple parameters like epsilon, dataset size etc. 
	"""
	def benchmark(self, benchmark_params : BenchmarkParams) -> BenchmarkMetrics:
		"""
		Benchmarks properties of privacy algorithm DP implementations using metrics
			- Privacy Promise
			- Accuracy Promise
			- Utility Promise
			- Bias Promise
		
		Returns a benchmark metrics object
		"""
		benchmark_res = BenchmarkMetrics()
        
		return benchmark_res