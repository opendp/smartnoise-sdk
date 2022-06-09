import pandas as pd
import numpy as np

from snsynth.pytorch.nn import DPCTGAN, PATECTGAN
from snsynth.preprocessors import DPSSTransformer

eps = 3.0
batch_size = 20
size = 100
pd_data = pd.DataFrame(columns=["A"], data=np.array(np.arange(0.0, size)).T)
np_data = np.array(np.arange(0.0, size)).astype(np.double)


class TestInputChecks:
    def test_input_checks_PATECTGAN_np(self):
        synth = PATECTGAN(epsilon=eps, batch_size=batch_size)
        try:
            synth.train(np_data, categorical_columns=[0], transformer=DPSSTransformer)
        except ValueError as v:
            assert str(v).startswith("It looks like")
            return
        raise AssertionError("DPCTGAN should have raised a ValueError")

    def test_input_checks_PATECTGAN_pd(self):
        synth = PATECTGAN(epsilon=eps, batch_size=batch_size)
        try:
            synth.train(pd_data, categorical_columns=["A"], transformer=DPSSTransformer)
        except ValueError as v:
            assert str(v).startswith("It looks like")
            return
        raise AssertionError("DPCTGAN should have raised a ValueError")

    def test_input_checks_DPCTGAN_np(self):
        synth = DPCTGAN(epsilon=eps, batch_size=batch_size)
        try:
            synth.train(np_data, categorical_columns=[0], transformer=DPSSTransformer)
        except ValueError as v:
            assert str(v).startswith("It looks like")
            return
        raise AssertionError("DPCTGAN should have raised a ValueError")

    def test_input_checks_DPCTGAN_pd(self):
        synth = DPCTGAN(epsilon=eps, batch_size=batch_size)
        try:
            synth.train(pd_data, categorical_columns=["A"], transformer=DPSSTransformer)
        except ValueError as v:
            assert str(v).startswith("It looks like")
            return
        raise AssertionError("DPCTGAN should have raised a ValueError")
