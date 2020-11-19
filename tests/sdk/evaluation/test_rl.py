import numpy as np
import pandas as pd
import csv
import logging
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.learner.q_learning import QLearning
from opendp.smartnoise.evaluation.learner.util import generate_query

test_logger = logging.getLogger("test-logger")


class TestQlearning():
    def test_qlearning(self):
        lp = LearnerParams(observation_space=30000, num_episodes=2, num_steps=2)
        b = QLearning(lp)
        select_path = os.path.join(os.path.dirname(__file__), "select.cfg")
        querypool = generate_query(2, select_path)
        b.qlearning(["SELECT COUNT(UserId) AS UserCount FROM dataset.dataset"] + querypool)
        output = b.qlearning(querypool)
        assert((output[0]['dpresult'] == 'DP_PASS') | (output[0]['dpresult'] == 'ActionResultedSameQuery') | (output[0]['dpresult'] == 'DP_BUG') | (output[0]['dpresult'] == 'ActionnotValid_ASTnotAvailable'))

