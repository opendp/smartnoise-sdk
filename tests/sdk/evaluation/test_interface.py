import logging
test_logger = logging.getLogger("eval-interface-test-logger")
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from dp_lib import DPSampleLibrary
from dp_algorithm import DPSample
import pandas as pd
import random
import pytest

class TestEval:
    def test_dp_algorithm(self):
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
        report = dv.release(df, actual=True)
        test_logger.debug("Actual count response: "  + str(report.res[firstkey]))

        assert(isinstance(report.res[firstkey], (int, float)))