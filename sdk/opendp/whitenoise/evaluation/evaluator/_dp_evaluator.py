from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from opendp.whitenoise.evaluation.evaluator._base import Evaluator

class DPEValuator(Evaluator):
    """
    Implement the Evaluator interface that takes in two neighboring datasets
    D1 and D2 and a privacy algorithm. Then runs the algorithm on the 
    two datasets to find whether that algorithm adheres to the privacy promise.
    """
    def evaluate(self, 
		d1 : object, 
		d2 : object, 
		algorithm : PrivacyAlgorithm, 
		privacy_params : PrivacyParams, 
		eval_params : EvaluatorParams) -> Metrics:
        """
		Evaluates properties of privacy algorithm DP implementations using 
			- DP Histogram Test
			- Accuracy Test
			- Utility Test
			- Bias Test
		
		d1 and d2 are neighboring datasets
		algorithm is the DP implementation object
		Returns a metrics object
		"""
        # Prepare the algorithm
        # TBD
        return