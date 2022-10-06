import numpy as np
import pandas as pd

from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import PATECTGAN, PATEGAN

size = 100
batch_size = 10
eps = 1.0

np_data_xy = (
    np.array([np.arange(0, size) % 3, (np.arange(0, size) % 3) * 10]).astype(np.int16).T
)


class TestDPGANInputChecks:
    def test_train_patectgan_continuous(self):
        dpgan = PATECTGAN(epsilon=eps, batch_size=batch_size)
        try:
            dpgan.train(np_data_xy, categorical_columns=[0, 1])
        except ValueError:
            return

    # raise AssertionError('PATECTGAN should have raised a ValueError')

    def test_train_pategan_continuous(self):
        dpgan = PATEGAN(epsilon=eps, batch_size=batch_size)
        try:
            dpgan.train(np_data_xy, categorical_columns=[0])
        except ValueError:
            return
        raise AssertionError("PATECTGAN should have raised a ValueError")

    def test_fit_pytorchgan_continuous(self):
        dpgan = PytorchDPSynthesizer(eps, PATECTGAN(epsilon=eps, batch_size=batch_size))
        pd_data_xy = pd.DataFrame(np_data_xy, columns=["x", "y"])
        try:
            dpgan.fit(
                pd_data_xy, categorical_columns=["x"]
            )
        except ValueError:
            return

    def test_fit_pytorchgan_continuous_pategan(self):
        dpgan = PytorchDPSynthesizer(
            eps, PATEGAN(epsilon=eps, batch_size=batch_size), None
        )
        pd_data_xy = pd.DataFrame(np_data_xy, columns=["x", "y"])
        try:
            dpgan.fit(pd_data_xy, categorical_columns=["x"])
        except ValueError:
            return
        raise AssertionError("PATECTGAN should have raised a ValueError")
