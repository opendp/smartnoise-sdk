from opendp.smartnoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.metrics._metrics import Metrics
from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
	Interface for various DP implementations to interface with tests available
	in evaluator. Evaluator tests for various properties of DP implementation
	like privacy, accuracy, utility and bias
	"""

    @abstractmethod
    def evaluate(
        self,
        d1: object,
        d2: object,
        pa: PrivacyAlgorithm,
        algorithm: object,
        privacy_params: PrivacyParams,
        eval_params: EvaluatorParams,
    ) -> {str: Metrics}:
        """
		Evaluates properties of privacy algorithm DP implementations using
			- DP Histogram Test
			- Accuracy Test
			- Utility Test
			- Bias Test

		d1 and d2 are neighboring datasets
		algorithm is the DP implementation object
		actual is the non-private implementation object
		Returns a metrics object
		"""
        pass
