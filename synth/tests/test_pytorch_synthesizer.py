import subprocess
import os

import pytest
import string
import pandas as pd

from snsynth.preprocessors import GeneralTransformer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import DPGAN, DPCTGAN, PATECTGAN

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

df = pd.read_csv(csv_path)
df_non_continuous = df[['sex','educ','race','married']]

nf = df.to_numpy()
nf_non_continuous = df_non_continuous.to_numpy()

@pytest.mark.torch
class TestPytorchDPSynthesizer_DPGAN:
    def setup(self):
        self.dpgan = PytorchDPSynthesizer(1.0, DPGAN(), GeneralTransformer())

    def test_fit(self):
        self.dpgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        assert self.dpgan.gan.generator

    def test_sample(self):
        self.dpgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df_non_continuous)
        synth_data = self.dpgan.sample(sample_size)
        assert synth_data.shape == df_non_continuous.shape

    def test_fit_continuous(self):
        dpgan = DPGAN(epsilon=1.0)
        df_continuous = df[['age','educ','income']]
        dpgan.train(df_continuous)
        synth_data = dpgan.generate(len(df_continuous))
        assert synth_data.shape == df_continuous.shape

class TestPytorchDPSynthesizer_DPCTGAN:
    def setup(self):
        self.dpctgan = PytorchDPSynthesizer(1.0, DPCTGAN(), None)

    def test_fit(self):
        df_non_continuous = df[['sex','educ','race','married']]
        self.dpctgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        assert self.dpctgan.gan._generator

    def test_sample(self):
        self.dpctgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.dpctgan.sample(sample_size)
        assert synth_data.shape == df_non_continuous.shape
    
    def test_fit_numpy(self):
        dpctgan = DPCTGAN(epsilon=1.0)
        dpctgan.train(nf_non_continuous, categorical_columns=[0, 1, 2, 3])

class TestPytorchDPSynthesizer_PATECTGAN:
    def setup(self):
        self.patectgan = PytorchDPSynthesizer(1.0, PATECTGAN(), None)

    def test_fit(self):
        self.patectgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        assert self.patectgan.gan._generator

    def test_sample(self):
        self.patectgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.patectgan.sample(sample_size)
        assert synth_data.shape == df_non_continuous.shape

class TestPytorchDPSynthesizer_PATECTDRAGAN:
    def setup(self):
        self.patectgan = PytorchDPSynthesizer(1.0, PATECTGAN(regularization='dragan'), None)

    def test_fit(self):
        self.patectgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        assert self.patectgan.gan._generator

    def test_sample(self):
        self.patectgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.patectgan.sample(sample_size)
        assert synth_data.shape == df_non_continuous.shape
