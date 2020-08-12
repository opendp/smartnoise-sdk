from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
import opendp.whitenoise.core as wn

class DPCore(PrivacyAlgorithm):
    """
    Sample implementation of PrivacyAlgorithm Interface
    that allows for the library to be stochastically tested by
    evaluator. 
    """
    def prepare(self, algorithm : object, privacy_params: PrivacyParams, eval_params: EvaluatorParams):
        """
        Load the algorithm (in this case OpenDP Core) to be used for acting on the dataset
        Initialize the privacy params that need to be used by the function
        for calculating differentially private noise
        """
        self.algorithm = algorithm
        self.privacy_params = privacy_params
        self.eval_params = eval_params

    def release(self, dataset: object, actual = False) -> Report:
        """
        Releases report according to the OpenDP Core applying 
        functions on the dataset or return the actual report
        if actual is set to True
        """
        if(not actual):
            noisy_res = {"__key__" : []}
            # Repeating analysis multiple times to collect enough samples for evaluation
            for i in range(self.eval_params.repeat_count):
                with wn.Analysis() as analysis:
                    data = wn.to_float(wn.Dataset(value=dataset))
                    agg = self.algorithm(
                        data=data,
                        privacy_usage={'epsilon': self.privacy_params.epsilon},
                        data_lower=float(min(dataset)),
                        data_upper=float(max(dataset)),
                        data_rows=len(dataset),
                        data_columns=1
                    )
                    analysis.release()
                    noisy_res["__key__"].append(agg.value)
            return Report(noisy_res)
        else:
            actual_res = {"__key__" : sum(dataset) / len(dataset)}
            return Report(actual_res)