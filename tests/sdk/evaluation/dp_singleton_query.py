from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.reader.rowset import TypedRowset
from opendp.whitenoise.sql import PrivateReader

class DPSingletonQuery(PrivacyAlgorithm):
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

    def release(self, dataset: object, actual = None) -> Report:
        """
        Dataset is a collection of [Dataset Metadata, PandasReader]
        Releases response to SQL query based on the number of repetitions
        requested by eval_params if actual is set of False. 
        Actual response is only returned once
        """
        if(not actual):
            private_reader = PrivateReader(dataset[0], dataset[1], self.privacy_params.epsilon)
            query_ast = private_reader.parse_query_string(self.algorithm)
            srs_orig = private_reader.reader.execute_ast_typed(query_ast)
            noisy_values = []
            for idx in range(self.eval_params.repeat_count):
                srs = TypedRowset(srs_orig.rows(), list(srs_orig.types.values()))
                res = private_reader._execute_ast(query_ast, True)
                noisy_values.append(res.rows()[1:][0][0])
            return Report({"__key__" : noisy_values})
        else:
            reader = dataset[1]
            exact = reader.execute_typed(actual).rows()[1:][0][0]
            return Report({"__key__" : exact})