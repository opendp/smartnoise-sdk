import logging
test_logger = logging.getLogger("eval-interface-test-logger")
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from opendp.whitenoise.evaluation.blackbox._base import BlackBoxPrivacyInterface
from dp_lib import DPSampleLibrary
from dp_blackbox import DPSampleInterface
import pandas as pd
import random
import pytest

class TestEvalInterface:
    def test_dp_blackbox(self):
        logging.getLogger().setLevel(logging.DEBUG)
        lib = DPSampleLibrary()
        dv = DPSampleInterface()
        pp = PrivacyParams(epsilon=1.0)
        ev = EvaluatorParams(repeat_count=500)
        df = pd.DataFrame(random.sample(range(1, 1000), 100), columns = ['Usage'])

        # Preparing and releasing from Sample DP library to send noisy results to evaluator
        dv.prepare(lib, pp, ev)
        report = dv.release(df)
        test_logger.debug("Repeated noisy count responses: "  + str(report.res_df.shape[0]))
        test_logger.debug("Count of Numerical columns in DP repeated response: "  + str(len(report.num_cols)))
        test_logger.debug("Count of Dimension columns in DP repeated response: "  + str(len(report.dim_cols)))
        assert(report.res_df.shape[0] == ev.repeat_count)
        assert(len(report.num_cols) == 1)
        assert(len(report.dim_cols) == 1)