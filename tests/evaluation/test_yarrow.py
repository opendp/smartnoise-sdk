import logging
test_logger = logging.getLogger("yarrow-test-logger")

import sys
import subprocess
import os
import pytest
import pandas as pd
from opendp.whitenoise.evaluation.dp_verification import DPVerification
from opendp.whitenoise.evaluation.exploration import Exploration
from opendp.whitenoise.evaluation.aggregation import Aggregation
import whitenoise
import whitenoise.components as op

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
dv = DPVerification(dataset_size=1000, csv_path=os.path.join(root_url, "service", "datasets"))
test_csv_path = os.path.join(root_url, "service", "datasets", "evaluation", "PUMS_1000.csv")
test_csv_names = ["age", "sex", "educ", "race", "income", "married"]

df = pd.read_csv(test_csv_path)
actual_mean = df['race'].mean()
actual_var = df['educ'].var()
actual_moment = df['race'].skew()
actual_covariance = df['age'].cov(df['married'])

class TestYarrow:
    def test_dp_yarrow_mean_pums(self):
        logging.getLogger().setLevel(logging.DEBUG)
        dp_res, bias_res = dv.yarrow_test(test_csv_path, test_csv_names, op.dp_mean, 'race', "FLOAT", epsilon=.65, actual = actual_mean, data_min=0., data_max=100., data_n=1000)
        test_logger.debug("Result of DP Predicate Test on Yarrow Mean: " + str(dp_res))
        test_logger.debug("Result of Bias Test on Yarrow Mean: " + str(bias_res))
        assert(dp_res == True)
        assert(bias_res == True)

    def test_dp_yarrow_var_pums(self):
        logging.getLogger().setLevel(logging.DEBUG)
        dp_res, bias_res = dv.yarrow_test(test_csv_path, test_csv_names, op.dp_variance, 'educ', "FLOAT", epsilon=.15, actual = actual_var, data_min=0., data_max=12., data_n=1000)
        test_logger.debug("Result of DP Predicate Test on Yarrow Mean: " + str(dp_res))
        test_logger.debug("Result of Bias Test on Yarrow Mean: " + str(bias_res))
        assert(dp_res == True)
        assert(bias_res == True)

    def test_dp_yarrow_moment_pums(self):
        logging.getLogger().setLevel(logging.DEBUG)
        dp_res, bias_res = dv.yarrow_test(test_csv_path, test_csv_names, op.dp_moment_raw, 'race', "FLOAT", epsilon=.15, actual = actual_moment, data_min=0., data_max=100., data_n=1000, order = 3)
        test_logger.debug("Result of DP Predicate Test on Yarrow Mean: " + str(dp_res))
        test_logger.debug("Result of Bias Test on Yarrow Mean: " + str(bias_res))
        assert(dp_res == True)
        assert(bias_res == True)

    def test_dp_yarrow_covariance_pums(self):
        logging.getLogger().setLevel(logging.DEBUG)
        dp_res, bias_res = dv.yarrow_test(test_csv_path, test_csv_names, op.dp_covariance, 'age', 'married', "FLOAT", actual = actual_covariance, epsilon=.15, left_n=1000, right_n=1000,left_min=0.,left_max=1.,right_min=0.,right_max=1.)
        test_logger.debug("Result of DP Predicate Test on Yarrow Mean: " + str(dp_res))
        test_logger.debug("Result of Bias Test on Yarrow Mean: " + str(bias_res))
        assert(dp_res == True)
        assert(bias_res == True)
