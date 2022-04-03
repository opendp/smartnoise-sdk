import numpy as np
from snsql.sql._mechanisms import *
from snsql.sql.privacy import Privacy, Stat
import pytest

# grid of (alpha, epsilon, delta, max_contrib) to test
grid = [
    (0.01, 0.5, 0.0, 3),
    (0.01, 0.1, 1/3333, 1),
    (0.05, 1.0, 0.0, 2),
    (0.25, 1.3, 1/1000, 3)
]


class TestSimpleAccuracy:
    def test_geom_count(self, test_databases):
        query = 'SELECT COUNT(educ) FROM PUMS.PUMS'
        sensitivity = 1
        for alpha, epsilon, delta, max_contrib in grid:
            privacy = Privacy(epsilon=epsilon, delta=delta)
            reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy, overrides={'max_contrib': max_contrib})
            if reader:
                mech_class = privacy.mechanisms.get_mechanism(sensitivity, 'count', 'int')
                mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
                assert(mech.mechanism == Mechanism.geometric)
                acc = reader.get_simple_accuracy(query, alpha)
                assert(np.isclose(acc[0], mech.accuracy(alpha)))
    def test_geom_small_sum(self, test_databases):
        query = 'SELECT SUM(age) FROM PUMS.PUMS'
        sensitivity = 100
        for alpha, epsilon, delta, max_contrib in grid:
            privacy = Privacy(epsilon=epsilon, delta=delta)
            reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy, overrides={'max_contrib': max_contrib})
            if reader:
                mech_class = privacy.mechanisms.get_mechanism(sensitivity, 'sum', 'int')
                mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
                assert(mech.mechanism == Mechanism.geometric)
                acc = reader.get_simple_accuracy(query, alpha)
                assert(np.isclose(acc[0], mech.accuracy(alpha)))
    def test_geom_large_sum(self, test_databases):
        # reverts to laplace because it's large
        query = 'SELECT SUM(income) FROM PUMS.PUMS'
        sensitivity = 500_000
        for alpha, epsilon, delta, max_contrib in grid:
            if delta == 0.0:
                delta = 1/100_000
            privacy = Privacy(epsilon=epsilon, delta=delta)
            reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy, overrides={'max_contrib': max_contrib})
            if reader:
                mech_class = privacy.mechanisms.get_mechanism(sensitivity, 'sum', 'int')
                mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
                assert(mech.mechanism == Mechanism.laplace)
                acc = reader.get_simple_accuracy(query, alpha)
                assert(np.isclose(acc[0], mech.accuracy(alpha)))
    def test_geom_key_count(self, test_databases):
        # reverts to laplace because we need a threshold
        query = 'SELECT COUNT(DISTINCT pid) FROM PUMS.PUMS'
        sensitivity = 1
        for alpha, epsilon, delta, max_contrib in grid:
            if delta == 0.0: # not permitted when thresholding
                delta = 1/100_000
            privacy = Privacy(epsilon=epsilon, delta=delta)
            reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy, overrides={'max_contrib': max_contrib})
            if reader:
                mech_class = privacy.mechanisms.get_mechanism(sensitivity, 'threshold', 'int')
                mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
                assert(mech.mechanism == Mechanism.laplace)
                acc = reader.get_simple_accuracy(query, alpha)
                assert(np.isclose(acc[0], mech.accuracy(alpha)))
    # def test_geom_key_count_gauss(self, test_databases):
    #     # reverts to gaussian because we need a threshold
    #     query = 'SELECT COUNT(DISTINCT pid) FROM PUMS.PUMS'
    #     sensitivity = 1
    #     for alpha, epsilon, delta, max_contrib in grid:
    #         if delta == 0.0: # not permitted when thresholding
    #             delta = 1/100_000
    #         privacy = Privacy(epsilon=epsilon, delta=delta)
    #         privacy.mechanisms.map[Stat.threshold] = Mechanism.gaussian
    #         reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy, overrides={'max_contrib': max_contrib})
    #         if reader:
    #             mech_class = privacy.mechanisms.get_mechanism(sensitivity, 'threshold', 'int')
    #             mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
    #             assert(mech.mechanism == Mechanism.gaussian)
    #             acc = reader.get_simple_accuracy(query, alpha)
    #             assert(np.isclose(acc[0], mech.accuracy(alpha)))
    # def test_gauss_count(self, test_databases):
    #     query = 'SELECT COUNT(educ) FROM PUMS.PUMS'
    #     sensitivity = 1
    #     for alpha, epsilon, delta, max_contrib in grid:
    #         if delta == 0.0:
    #             delta = 1/100_000
    #         privacy = Privacy(epsilon=epsilon, delta=delta)
    #         privacy.mechanisms.map[Stat.count] = Mechanism.gaussian
    #         reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy, overrides={'max_contrib': max_contrib})
    #         if reader:
    #             mech_class = privacy.mechanisms.get_mechanism(sensitivity, 'count', 'int')
    #             mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
    #             assert(mech.mechanism == Mechanism.gaussian)
    #             acc = reader.get_simple_accuracy(query, alpha)
    #             assert(np.isclose(acc[0], mech.accuracy(alpha)))
    # def test_lap_count(self, test_databases):
    #     query = 'SELECT COUNT(educ) FROM PUMS.PUMS'
    #     sensitivity = 1
    #     for alpha, epsilon, delta, max_contrib in grid:
    #         if delta == 0.0:
    #             delta = 1/100_000
    #         privacy = Privacy(epsilon=epsilon, delta=delta)
    #         privacy.mechanisms.map[Stat.count] = Mechanism.laplace
    #         reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy, overrides={'max_contrib': max_contrib})
    #         if reader:
    #             mech_class = privacy.mechanisms.get_mechanism(sensitivity, 'count', 'int')
    #             mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
    #             assert(mech.mechanism == Mechanism.laplace)
    #             acc = reader.get_simple_accuracy(query, alpha)
    #             assert(np.isclose(acc[0], mech.accuracy(alpha)))

class TestSimpleMatch:
    """
    Compare accuracies obtained from the reader without executing query
    with accuracies provided inline with query result.
    """
    def test_simple_pid(self, test_databases):
        max_ids = 2
        alpha = 0.05
        privacy = Privacy(alphas=[alpha], epsilon=1.5, delta=1/100_000)
        privacy.mechanisms.map[Stat.threshold] = Mechanism.gaussian
        query = 'SELECT COUNT(DISTINCT pid), COUNT(*), COUNT(educ), SUM(age) FROM PUMS.PUMS'
        reader = test_databases.get_private_reader(
            database='PUMS_pid', 
            engine="pandas", 
            privacy=privacy,
            overrides={'max_ids': max_ids})
        if reader:
            simple_a = reader.get_simple_accuracy(query, alpha=alpha)
            res = reader.execute(query, accuracy=True)
            simple_b = res[1][1][0]
            assert (all([a == b for a, b in zip(simple_a, simple_b)]))
    def test_simple_row_privacy(self, test_databases):
        alpha = 0.07
        privacy = Privacy(alphas=[alpha], epsilon=0.5, delta=1/1000)
        query = 'SELECT COUNT(*), COUNT(educ), SUM(age) FROM PUMS.PUMS'
        reader = test_databases.get_private_reader(
            database='PUMS', 
            engine="pandas", 
            privacy=privacy
        )
        if reader:
            simple_a = reader.get_simple_accuracy(query, alpha=alpha)
            res = reader.execute(query, accuracy=True)
            simple_b = res[1][1][0]
            assert (all([a == b for a, b in zip(simple_a, simple_b)]))

