from opendp.smartnoise.evaluation.metrics._metrics import Metrics
from opendp.smartnoise.evaluation.params._halton_params import HaltonParams
from abc import ABC, abstractmethod

class Explorer(ABC):
	"""
	DP evaluator can be invoked with various evaluation parameters
	For example, for a SQL analysis, we can pass various datasets and
	queries to see if the evaluator metrics are successful. This interface
	helps provide capability to do brute force generation of neighboring
	datasets.
	"""
	@abstractmethod
	def evaluate_powerset(self, dataset : object) -> Metrics:
		"""
		Explores powerset of a given dataset
		"""
		pass

	@abstractmethod
	def generate_halton(self, halton_params : HaltonParams) -> [object]:
		"""
		Generate new datasets using halton sequence. Calls the powerset explore
		"""
		pass
