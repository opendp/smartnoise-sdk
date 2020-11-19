import logging
import numpy as np
import pandas as pd
test_logger = logging.getLogger("test-logger")

from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.learner._generate import Grammar
from opendp.smartnoise.evaluation.learner.util import create_simulated_dataset, generate_neighbors
from opendp.smartnoise.evaluation.learner import bandit

from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.params._dataset_params import DatasetParams
from opendp.smartnoise.evaluation.evaluator._dp_evaluator import DPEvaluator
from opendp.smartnoise.sql import PandasReader
from dp_singleton_query import DPSingletonQuery 


class TestBandit():
    def setup(self):
        self.pp = PrivacyParams(epsilon=1.0)
        self.ev = EvaluatorParams(repeat_count=100)
        self.dd = DatasetParams(dataset_size=500)
        self.pa = DPSingletonQuery()


    def bandit(self, query):
        df, metadata = create_simulated_dataset(self.dd.dataset_size, "dataset")
        d1_dataset, d2_dataset, d1_metadata, d2_metadata = generate_neighbors(df, metadata)
        d1 = PandasReader(d1_dataset, d1_metadata)
        d2 = PandasReader(d2_dataset, d2_metadata)
        eval = DPEvaluator()
        pa = DPSingletonQuery()
        key_metrics = eval.evaluate([d1_metadata, d1], [d2_metadata, d2], pa, query, self.pp, self.ev)
        for key, metrics in key_metrics.items():
            dp_res = metrics.dp_res
            js_res = metrics.jensen_shannon_divergence
            assert(metrics.dp_res == True)
            assert(metrics.jensen_shannon_divergence > 0.0)
            test_logger.debug("Wasserstein Distance:" + str(metrics.wasserstein_distance))
            test_logger.debug("Jensen Shannon Divergence:" + str(metrics.jensen_shannon_divergence))

    def test_bandit(self):
        query= "SELECT COUNT(UserId) AS UserCount FROM dataset.dataset"
        self.bandit(query)
