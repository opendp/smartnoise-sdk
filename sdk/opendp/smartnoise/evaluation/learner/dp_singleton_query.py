from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.report._report import Report
from opendp.smartnoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.smartnoise.reader.rowset import TypedRowset
from opendp.smartnoise.sql import PrivateReader

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

    def release(self, dataset: object) -> Report:
        """
        Dataset is a collection of [Dataset Metadata, PandasReader]
        Releases response to SQL query based on the number of repetitions
        requested by eval_params if actual is set of False. 
        
        """
        private_reader = PrivateReader(dataset[0], dataset[1], self.privacy_params.epsilon)
        query_ast = private_reader.parse_query_string(self.algorithm)
        srs_orig = private_reader.reader.execute_ast_typed(query_ast)
        noisy_values = []
        for idx in range(self.eval_params.repeat_count):
            srs = TypedRowset(srs_orig.rows(), list(srs_orig.types.values()))
            res = private_reader._execute_ast(query_ast, True)
            try:
                noisy_values.append(res.rows()[1:][0][0])
            except Exception as e: 
                return Report({"__key__" : "noisy_values_empty, " + str(type(e))+ ", "+str(e)})
        return Report({"__key__" : noisy_values})

    def actual_release(self, dataset):
        """
        Return exact (non-private) query response. 
        Exact response is only returned once
        """
        reader = dataset[1]
        try:
            exact = reader.execute_typed(self.algorithm).rows()[1:][0][0]
        except Exception as e: 
            return Report({"__key__" : "exact_value_error, " + str(type(e))+ ", " + str(e)})
        return Report({"__key__" : exact})
