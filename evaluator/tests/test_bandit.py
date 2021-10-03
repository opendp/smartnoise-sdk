import logging
import os
import numpy as np
import pandas as pd

from sneval.params._learner_params import LearnerParams
from sneval.learner._generate import Grammar
from sneval.learner.util import create_simulated_dataset, generate_neighbors
from sneval.learner import bandit

from sneval.params._learner_params import LearnerParams
from sneval.params._privacy_params import PrivacyParams
from sneval.params._eval_params import EvaluatorParams
from sneval.params._dataset_params import DatasetParams
from sneval.evaluator._dp_evaluator import DPEvaluator
from snsql.sql import PandasReader
from dp_singleton_query import DPSingletonQuery
from sneval.learner.bandit import Bandit
from sneval.learner.util import generate_query

test_logger = logging.getLogger("test-logger")


class TestBandit():
    def test_bandit(self):
        b = Bandit()
        select_path = os.path.join(os.path.dirname(__file__), "select.cfg")
        querypool = generate_query(3, select_path)
        output = b.learn(["SELECT COUNT(UserId) AS UserCount FROM dataset.dataset"] + querypool)
        assert(output[0]['dpresult'] == True)
        assert(output[0]['jensen_shannon_divergence'] > 0.0)





