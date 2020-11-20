import numpy as np
import pandas as pd 
import csv
import logging
test_logger = logging.getLogger("test-logger")
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.params._dataset_params import DatasetParams
from opendp.smartnoise.evaluation.learner.Qlearning import QLearning
from opendp.smartnoise.evaluation.learner.util import generate_query

class TestQlearning():
    def test_qlearning(self):
        lp = LearnerParams(observation_space=30000, num_episodes=2, num_steps=2)
        b = QLearning(lp, PrivacyParams, EvaluatorParams, DatasetParams)
        querypool = generate_query(2)
        b.qlearning(["SELECT COUNT(UserId) AS UserCount FROM dataset.dataset"] + querypool)
        output = b.qlearning(querypool)
        assert((output[0]['dpresult'] == 'DP_PASS') | (output[0]['dpresult'] == 'ActionResultedSameQuery') | (output[0]['dpresult'] == 'DP_BUG') | (output[0]['dpresult'] == 'ActionnotValid_ASTnotAvailable'))

