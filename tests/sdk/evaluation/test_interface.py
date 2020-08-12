import logging
test_logger = logging.getLogger("eval-interface-test-logger")
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.params._benchmark_params import BenchmarkParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.evaluator._dp_evaluator import DPEvaluator
from opendp.whitenoise.evaluation.benchmarking._dp_benchmark import DPBenchmarking
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from dp_lib import DPSampleLibrary
from dp_algorithm import DPSample
from dp_multikey import DPMultiKey
import pandas as pd
import numpy as np
import random
import pytest

class TestEval:
    def test_interface_algorithm(self):
        logging.getLogger().setLevel(logging.DEBUG)
        lib = DPSampleLibrary()
        dv = DPSample()
        pp = PrivacyParams(epsilon=1.0)
        ev = EvaluatorParams(repeat_count=500)
        df = pd.DataFrame(random.sample(range(1, 1000), 100), columns = ['Usage'])

        # Preparing and releasing from Sample DP algorithm to send noisy results to evaluator
        dv.prepare(lib.dp_count, pp, ev)
        report = dv.release(df)
        
        # Test DP respose from interface
        assert(isinstance(report.res,  dict))
        assert(len(report.res) > 0)
        firstkey = list(report.res.keys())[0]
        test_logger.debug("First key name is:" + str(firstkey))
        test_logger.debug("Repeated noisy count responses: "  + str(report.res[firstkey]))
        assert(isinstance(firstkey, str))
        assert(len(report.res[firstkey]) == ev.repeat_count)

        # Test non-DP i.e. actual response from interface should be a single numeric return
        report = dv.release(df, actual=len)
        test_logger.debug("Actual count response: "  + str(report.res[firstkey]))

        assert(isinstance(report.res[firstkey], (int, float)))

    def test_interface_evaluator(self):
        logging.getLogger().setLevel(logging.DEBUG)
        lib = DPSampleLibrary()
        pa = DPSample()
        metrics = Metrics()
        # Before running the DP test, it should be default to False
        # and Wasserstein distance should be 0 
        assert(metrics.dp_res == False)
        assert(metrics.wasserstein_distance == 0.0)
        assert(metrics.jensen_shannon_divergence == 0.0)
        assert(metrics.kl_divergence == 0.0)
        assert(metrics.mse == 0.0)
        assert(metrics.std == 0.0)
        assert(metrics.msd == 0.0)
        pp = PrivacyParams(epsilon=1.0)
        ev = EvaluatorParams(repeat_count=500)
        # Creating neighboring datasets
        d1 = pd.DataFrame(random.sample(range(1, 1000), 100), columns = ['Usage'])
        drop_idx = np.random.choice(d1.index, 1, replace=False)
        d2 = d1.drop(drop_idx)
        # Call evaluate
        eval = DPEvaluator()
        key_metrics = eval.evaluate(d1, d2, pa, lib.dp_count, len, pp, ev)
        # After evaluation, it should return True and distance metrics should be non-zero
        for key, metrics in key_metrics.items():
            assert(metrics.dp_res == True)
            test_logger.debug("Wasserstein Distance:" + str(metrics.wasserstein_distance))
            test_logger.debug("Jensen Shannon Divergence:" + str(metrics.jensen_shannon_divergence))
            test_logger.debug("KL Divergence:" + str(metrics.kl_divergence))
            test_logger.debug("MSE:" + str(metrics.mse))
            test_logger.debug("Standard Deviation:" + str(metrics.std))
            test_logger.debug("Mean Signed Deviation:" + str(metrics.msd))
            assert(metrics.wasserstein_distance > 0.0)
            assert(metrics.jensen_shannon_divergence > 0.0)
            assert(metrics.kl_divergence != 0.0)
            assert(metrics.mse > 0.0)
            assert(metrics.std != 0.0)
            assert(metrics.msd != 0.0)

    def test_interface_benchmark(self):
        logging.getLogger().setLevel(logging.DEBUG)
        lib = DPSampleLibrary()
        pa = DPSample()
        epsilon_list = [0.001, 0.5, 1.0, 2.0, 4.0]
        pp = PrivacyParams(epsilon=1.0)
        ev = EvaluatorParams(repeat_count=500)
        # Creating neighboring datasets
        d1 = pd.DataFrame(random.sample(range(1, 1000), 100), columns = ['Usage'])
        drop_idx = np.random.choice(d1.index, 1, replace=False)
        d2 = d1.drop(drop_idx)
        benchmarking = DPBenchmarking()
        # Preparing benchmarking params
        pa_algorithms = {pa : [lib.dp_count, len]}
        privacy_params_list = []
        for epsilon in epsilon_list:
            pp = PrivacyParams()
            pp.epsilon = epsilon
            privacy_params_list.append(pp)
        d1_d2_list = [[d1, d2]]
        benchmark_params = BenchmarkParams(pa_algorithms, privacy_params_list, d1_d2_list, ev)
        benchmark_metrics_list = benchmarking.benchmark(benchmark_params)
        for bm in benchmark_metrics_list:
            for key, metrics in bm.key_metrics.items():
                test_logger.debug("Epsilon: " + str(bm.privacy_params.epsilon) + \
                    " MSE:" + str(metrics.mse) + \
                    " Privacy Test: " + str(metrics.dp_res))
                assert(metrics.dp_res == True)
        assert(len(benchmark_metrics_list) == 5)

    def test_interface_multikey(self):
        logging.getLogger().setLevel(logging.DEBUG)
        lib = DPSampleLibrary()
        pa = DPMultiKey()
        metrics = Metrics()
        # Before running the DP test, it should be default to False
        # and Wasserstein distance should be 0 
        assert(metrics.dp_res == False)
        assert(metrics.wasserstein_distance == 0.0)
        assert(metrics.jensen_shannon_divergence == 0.0)
        assert(metrics.kl_divergence == 0.0)
        assert(metrics.mse == 0.0)
        assert(metrics.std == 0.0)
        assert(metrics.msd == 0.0)
        pp = PrivacyParams(epsilon=1.0)
        ev = EvaluatorParams(repeat_count=500)
        # Creating neighboring datasets
        col1 = list(range(0, 1000))
        col2 = list(range(-1000, 0))
        d1 = pd.DataFrame(list(zip(col1, col2)), columns=['Col1', 'Col2'])
        drop_idx = np.random.choice(d1.index, 1, replace=False)
        d2 = d1.drop(drop_idx)
        # Call evaluate
        eval = DPEvaluator()
        key_metrics = eval.evaluate(d1, d2, pa, lib.dp_sum, sum, pp, ev)
        # After evaluation, it should return True and distance metrics should be non-zero
        for key, metrics in key_metrics.items():
            assert(metrics.dp_res == True)
            test_logger.debug("Wasserstein Distance:" + str(metrics.wasserstein_distance))
            test_logger.debug("Jensen Shannon Divergence:" + str(metrics.jensen_shannon_divergence))
            test_logger.debug("KL Divergence:" + str(metrics.kl_divergence))
            test_logger.debug("MSE:" + str(metrics.mse))
            test_logger.debug("Standard Deviation:" + str(metrics.std))
            test_logger.debug("Mean Signed Deviation:" + str(metrics.msd))
            assert(metrics.wasserstein_distance > 0.0)
            assert(metrics.jensen_shannon_divergence > 0.0)
            assert(metrics.kl_divergence != 0.0)
            assert(metrics.mse > 0.0)
            assert(metrics.std != 0.0)
            assert(metrics.msd != 0.0)