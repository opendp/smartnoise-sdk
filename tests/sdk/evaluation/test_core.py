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
from dp_core import DPCore
import random
import pytest

class TestCore:
    @pytest.mark.skip(reason="Skipping in System build as it uses OpenDP Core")
    def test_interface_core_eval(self):
        """
        Testing interface of OpenDP Core        
        """
        import opendp.whitenoise.core as wn
        logging.getLogger().setLevel(logging.DEBUG)
        pa = DPCore()
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
        ev = EvaluatorParams(repeat_count=100)
        # Creating neighboring datasets
        d1 = list(range(1, 500))
        d2 = list(range(1, 500))
        drop_elem = random.choice(d2)
        d2.remove(drop_elem)
        # Call evaluate
        eval = DPEvaluator()
        key_metrics = eval.evaluate(d1, d2, pa, wn.dp_mean, pp, ev)
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

    @pytest.mark.skip(reason="Skipping in System build as it uses OpenDP Core")
    def test_interface_core_benchmark(self):
        """
        Benchmarks algorithm in OpenDP Core like DP Mean
        """
        import opendp.whitenoise.core as wn
        logging.getLogger().setLevel(logging.DEBUG)
        pa = DPCore()
        epsilon_list = [0.001, 0.5, 1.0, 2.0, 4.0]
        pp = PrivacyParams(epsilon=1.0)
        ev = EvaluatorParams(repeat_count=100)
        # Creating neighboring datasets
        d1 = list(range(1, 500))
        d2 = list(range(1, 500))
        drop_elem = random.choice(d2)
        d2.remove(drop_elem)
        benchmarking = DPBenchmarking()
        # Preparing benchmarking params
        pa_algorithms = {pa : wn.dp_mean}
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