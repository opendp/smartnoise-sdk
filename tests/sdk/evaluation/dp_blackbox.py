from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.blackbox._base import BlackBoxPrivacyInterface

class DPSampleInterface(BlackBoxPrivacyInterface):
    """
    Sample implementation of BlackBoxPrivacy Interface
    that allows for the library to be stochastically tested by
    evaluator. 
    """
    def prepare(self, analysis: object, privacy_params: PrivacyParams, eval_params: EvaluatorParams):
        """
        Load the function (analysis) to be used for acting on the dataset
        Initialize the privacy params that need to be used by the function
        for calculating differentially private noise
        """
        self.analysis = analysis
        self.privacy_params = privacy_params
        self.eval_params = eval_params

    def release(self, dataset: object, actual = False) -> Report:
        noisy_df = self.analysis.dp_count(dataset, self.privacy_params, self.eval_params)
        return Report(noisy_df)