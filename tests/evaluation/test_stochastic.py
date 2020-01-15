# This file contains stochastic tests for mechanism and SQL queries that should pass
# If these tests fail, then one of the 3 promises of DP system - Privacy, Accuracy, Utility are not being met

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pytest
from evaluation.DPVerification import DPVerification
from evaluation.Exploration import Exploration
from evaluation.Aggregation import Aggregation

dv = DPVerification(dataset_size=10000)
ag = Aggregation(t=1, repeat_count=10000)

class TestStochastic:
    def test_dp_predicate_count(self):
        d1_query = "SELECT COUNT(UserId) AS UserCount FROM d1.d1"
        d2_query = "SELECT COUNT(UserId) AS UserCount FROM d2.d2"
        dp_res, acc_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=10000)
        print("Result of DP Predicate Test on COUNT Query: ", dp_res)
        assert(dp_res == True)

    def test_dp_predicate_sum(self):
        d1_query = "SELECT SUM(Usage) AS TotalUsage FROM d1.d1"
        d2_query = "SELECT SUM(Usage) AS TotalUsage FROM d2.d2"
        dp_res, acc_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=10000)
        print("Result of DP Predicate Test on SUM Query: ", dp_res)
        assert(dp_res == True)    
    
    def test_dp_predicate_mean(self):
        d1_query = "SELECT AVG(Usage) AS MeanUsage FROM d1.d1"
        d2_query = "SELECT AVG(Usage) AS MeanUsage FROM d2.d2"
        dp_res, acc_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=10000)
        print("Result of DP Predicate Test on MEAN Query: ", dp_res)
        assert(dp_res == True)

    def test_dp_predicate_var(self):
        d1_query = "SELECT VAR(Usage) AS UsageVariance FROM d1.d1"
        d2_query = "SELECT VAR(Usage) AS UsageVariance FROM d2.d2"
        dp_res, acc_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=10000)
        print("Result of DP Predicate Test on VAR Query: ", dp_res)
        assert(dp_res == True)

    def test_dp_laplace_mechanism_count(self):
        dp_count, ks_count, ws_count = dv.aggtest(ag.dp_mechanism_count, 'UserId', binsize="auto", debug = False)
        assert(dp_count == True)

    def test_dp_laplace_mechanism_sum(self):
        dp_sum, ks_sum, ws_sum = dv.aggtest(ag.dp_mechanism_sum, 'Usage', binsize="auto", debug=False)
        assert(dp_sum == True)
    
    def test_dp_gaussian_mechanism_count(self):
        ag = Aggregation(t=1, repeat_count=10000, mechanism = "Gaussian")
        dp_count, ks_count, ws_count = dv.aggtest(ag.dp_mechanism_count, 'UserId', binsize="auto", debug = False)
        assert(dp_count == True)
    
    def test_dp_gaussian_mechanism_sum(self):
        ag = Aggregation(t=1, repeat_count=10000, mechanism = "Gaussian")
        dp_sum, ks_sum, ws_sum = dv.aggtest(ag.dp_mechanism_sum, 'Usage', binsize="auto", debug=False)
        assert(dp_sum == True)
