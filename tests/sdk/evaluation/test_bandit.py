import logging
import os
import numpy as np
import pandas as pd

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
from opendp.smartnoise.evaluation.learner.bandit import Bandit
from opendp.smartnoise.evaluation.learner.util import generate_query

test_logger = logging.getLogger("test-logger")


class TestBandit():
    def test_bandit(self):
        b = Bandit()
        select_path = os.path.join(os.path.dirname(__file__), "select.cfg")
        querypool = generate_query(3, select_path)
        output = b.learn(["SELECT COUNT(UserId) AS UserCount FROM dataset.dataset"] + querypool)
        assert(output[0]['dpresult'] == True)
        assert(output[0]['jensen_shannon_divergence'] > 0.0)





