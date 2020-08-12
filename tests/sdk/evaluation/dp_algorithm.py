from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm

class DPSample(PrivacyAlgorithm):
    """
    Sample implementation of PrivacyAlgorithm Interface
    that allows for the library to be stochastically tested by
    evaluator. 
    """
    def prepare(self, algorithm: object, privacy_params: PrivacyParams, eval_params: EvaluatorParams):
        """
        Load the algorithm to be used for acting on the dataset
        Initialize the privacy params that need to be used by the function
        for calculating differentially private noise
        """
        self.algorithm = algorithm
        self.privacy_params = privacy_params
        self.eval_params = eval_params

    def release(self, dataset: object, actual = None) -> Report:
        if(not actual):
            noisy_res = self.algorithm(dataset, self.privacy_params, self.eval_params)
            return Report(noisy_res)
        else:
            actual_res = {"__key__" : actual(dataset)}
            return Report(actual_res)