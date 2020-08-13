from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.reader.rowset import TypedRowset
from opendp.whitenoise.sql import PrivateReader

class DPMultiKey(PrivacyAlgorithm):
    """
    Sample implementation of PrivacyAlgorithm Interface
    that allows for the library to be stochastically tested by
    evaluator. 
    """
    def prepare(self, algorithm : object, privacy_params: PrivacyParams, eval_params: EvaluatorParams):
        """
        Load the algorithm (in this case SQL aggregation query) to be used for acting on the dataset
        Initialize the privacy params that need to be used by the function
        for calculating differentially private noise
        """
        self.algorithm = algorithm
        self.privacy_params = privacy_params
        self.eval_params = eval_params

    def release(self, dataset: object) -> Report:
        """
        Dataset is Pandas Dataframe with multiple columns and we need to sum
        elements in each column and assign a key (column name) for each column.  
        Releases count per key based on the number of repetitions
        requested by eval_params if actual is set of False. 
        Actual response is only returned once
        """
        noisy_res = self.algorithm(dataset, self.privacy_params, self.eval_params)
        return Report(noisy_res)

    def actual_release(self, dataset: object) -> Report:
        """
        Returns exact non-private response from algorithm
        """
        actual_res = self.algorithm(dataset, self.privacy_params, self.eval_params, actual = True)
        return Report(actual_res)