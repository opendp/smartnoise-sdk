from opendp.whitenoise.evaluation.blackbox._base import BlackBoxPrivacyInterface
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from abc import ABC, abstractmethod

class EvaluatorInterface(ABC):
	"""
	Interface for various DP implementations to interface with tests available
	in evaluator. Evaluator tests for various properties of DP implementation
	like privacy, accuracy, utility and bias
	"""
	@abstractmethod
	def evaluate(self, 
		d1 : object, 
		d2 : object, 
		analysis : BlackBoxPrivacyInterface, 
		privacy_params : PrivacyParams, 
		eval_params : EvaluatorParams) -> Metrics:
		"""
		Evaluates properties of black box DP implementations using 
			- DP Histogram Test
			- Accuracy Test
			- Utility Test
			- Bias Test
		
		d1 and d2 are neighboring datasets
		analysis is the DP implementation object
		Returns a metrics object
		"""
		pass