# This file contains stochastic tests for mechanism and SQL queries that should pass
# If these tests fail, then one of the 3 promises of DP system - Privacy, Accuracy, Utility are not being met

import logging
test_logger = logging.getLogger("stochastic-test-logger")

import sys
import os
import pytest
from evaluation.dp_verification import DPVerification
from evaluation.exploration import Exploration
from evaluation.aggregation import Aggregation
import yarrow

dv = DPVerification(dataset_size=1000)
ag = Aggregation(t=1, repeat_count=1000)

class TestStochastic:
    def test_dp_predicate_count(self):
        logging.getLogger().setLevel(logging.DEBUG)
        d1_query = "SELECT COUNT(UserId) AS UserCount FROM d1.d1"
        d2_query = "SELECT COUNT(UserId) AS UserCount FROM d2.d2"
        dp_res, acc_res, utility_res, bias_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=500)
        test_logger.debug("Result of DP Predicate Test on COUNT Query: " + str(dp_res))
        test_logger.debug("Result of Accuracy Test on COUNT Query: " + str(acc_res))
        assert(dp_res == True)
        assert(acc_res == True)
        assert(utility_res == True)
        assert(bias_res == True)

    def test_dp_predicate_sum(self):
        logging.getLogger().setLevel(logging.DEBUG)
        d1_query = "SELECT SUM(Usage) AS TotalUsage FROM d1.d1"
        d2_query = "SELECT SUM(Usage) AS TotalUsage FROM d2.d2"
        dp_res, acc_res, utility_res, bias_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=500)
        test_logger.debug("Result of DP Predicate Test on SUM Query: " + str(dp_res))
        test_logger.debug("Result of Accuracy Test on SUM Query: " + str(acc_res))
        assert(dp_res == True)
        assert(acc_res == True)
        assert(utility_res == True)
        assert(bias_res == True)
    
    def test_dp_predicate_mean(self):
        logging.getLogger().setLevel(logging.DEBUG)
        d1_query = "SELECT AVG(Usage) AS MeanUsage FROM d1.d1"
        d2_query = "SELECT AVG(Usage) AS MeanUsage FROM d2.d2"
        dp_res, acc_res, utility_res, bias_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=500)
        test_logger.debug("Result of DP Predicate Test on MEAN Query: " + str(dp_res))
        assert(dp_res == True)
        assert(bias_res == True)

    def test_dp_predicate_var(self):
        logging.getLogger().setLevel(logging.DEBUG)
        d1_query = "SELECT VAR(Usage) AS UsageVariance FROM d1.d1"
        d2_query = "SELECT VAR(Usage) AS UsageVariance FROM d2.d2"
        dp_res, acc_res, utility_res, bias_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=500, get_exact=False)
        test_logger.debug("Result of DP Predicate Test on VAR Query: " + str(dp_res))
        assert(dp_res == True)

    def test_dp_laplace_mechanism_count(self):
        dp_count, ks_count, ws_count = dv.aggtest(ag.dp_mechanism_count, 'UserId', binsize="auto", plot=False, debug = False)
        assert(dp_count == True)

    def test_dp_laplace_mechanism_sum(self):
        dp_sum, ks_sum, ws_sum = dv.aggtest(ag.dp_mechanism_sum, 'Usage', binsize="auto", plot=False, debug=False)
        assert(dp_sum == True)
    
    def test_dp_gaussian_mechanism_count(self):
        ag = Aggregation(t=1, repeat_count=500, mechanism = "Gaussian")
        dp_count, ks_count, ws_count = dv.aggtest(ag.dp_mechanism_count, 'UserId', binsize="auto", plot=False, debug = False)
        assert(dp_count == True)
    
    def test_dp_gaussian_mechanism_sum(self):
        ag = Aggregation(t=1, repeat_count=500, mechanism = "Gaussian")
        dp_sum, ks_sum, ws_sum = dv.aggtest(ag.dp_mechanism_sum, 'Usage', binsize="auto", plot=False, debug=False)
        assert(dp_sum == True)
    
    @pytest.mark.slow
    def test_powerset_sum(self):
        query_str = "SELECT SUM(Usage) AS TotalUsage FROM "
        dp_res, acc_res, utility_res, bias_res = dv.dp_powerset_test(query_str, repeat_count=500, plot=False)
        test_logger.debug("Result of DP Predicate Test on Powerset SUM: " + str(dp_res))
        test_logger.debug("Result of Accuracy Test on Powerset SUM: " + str(acc_res))
        assert(dp_res == True)
        assert(acc_res == True)
        assert(utility_res == True)
        assert(bias_res == True)

    def test_groupby(self):
        d1_query = "SELECT Role, Segment, COUNT(UserId) AS UserCount, SUM(Usage) AS Usage FROM d1.d1 GROUP BY Role, Segment"
        d2_query = "SELECT Role, Segment, COUNT(UserId) AS UserCount, SUM(Usage) AS Usage FROM d2.d2 GROUP BY Role, Segment"
        dp_res, acc_res, utility_res, bias_res = dv.dp_groupby_query_test(d1_query, d2_query, plot=False, repeat_count=500)
        test_logger.debug("Result of DP Predicate Test on GROUP BY: " + str(dp_res))
        test_logger.debug("Result of Accuracy Test on GROUP BY: " + str(acc_res))
        assert(dp_res == True)
        assert(acc_res == True)
        assert(utility_res == True)
        assert(bias_res == True)

    @pytest.mark.skip(reason="Yarrow response error while calling")
    def test_yarrow_dp_mean(self):
        test_csv_path = 'service/datasets/PUMS.csv'
        dp_yarrow_mean_res = dv.yarrow_test(test_csv_path, yarrow.dp_mean, 'income', float, epsilon=1.0, minimum=0, maximum=100, num_records=1000)
        assert(dp_yarrow_mean_res == True)

    @pytest.mark.skip(reason="Yarrow response error while calling")
    def test_yarrow_dp_variance(self):
        test_csv_path = 'service/datasets/PUMS.csv'
        dp_yarrow_var_res = dv.yarrow_test(test_csv_path, yarrow.dp_variance, 'educ', int, epsilon=1.0, minimum=0, maximum=12, num_records=1000)
        assert(dp_yarrow_var_res == True)

    @pytest.mark.skip(reason="Yarrow response error while calling")
    def test_yarrow_dp_moment_raw(self):
        test_csv_path = 'service/datasets/PUMS.csv'
        dp_yarrow_moment_res = dv.yarrow_test(test_csv_path, yarrow.dp_moment_raw, 'married', float, epsilon=.15, minimum=0, maximum=12, num_records=1000000, order = 3)
        assert(dp_yarrow_moment_res == True)

    @pytest.mark.skip(reason="Yarrow response error while calling")
    def test_yarrow_dp_covariance(self):
        test_csv_path = 'service/datasets/PUMS.csv'
        dp_yarrow_covariance_res = dv.yarrow_test(test_csv_path, yarrow.dp_covariance, 'married', int, 'sex', int, epsilon=.15, minimum_x=0, maximum_x=1, minimum_y=0, maximum_y=1, num_records=1000)
        assert(dp_yarrow_covariance_res == True)
