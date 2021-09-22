import subprocess
import os

import pytest
import string
import pandas as pd

from opendp.smartnoise.metadata import CollectionMetadata

try:
    from opendp.smartnoise.synthesizers.preprocessors import GeneralTransformer
    from opendp.smartnoise.synthesizers.pytorch import PytorchDPSynthesizer
    from opendp.smartnoise.synthesizers.pytorch.nn import DPGAN, DPCTGAN, PATECTGAN

except Exception as e:
    import logging
    test_logger = logging.getLogger(__name__)
    test_logger.warning("Requires torch and torchdp. Failed with Exception {}".format(e))


git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

@pytest.mark.torch
class TestPytorchDPSynthesizer_DPGAN:
    def setup(self):
        self.dpgan = PytorchDPSynthesizer(1.0, DPGAN(), GeneralTransformer())

    def test_fit(self):
        df_non_continuous = df[['sex','educ','race','married']]
        self.dpgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        assert self.dpgan.gan.generator

    def test_sample(self):
        df_non_continuous = df[['sex','educ','race','married']]
        self.dpgan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df_non_continuous)
        synth_data = self.dpgan.sample(sample_size)
        assert synth_data.shape == df_non_continuous.shape

class TestPytorchDPSynthesizer_DPCTGAN:
    def setup(self):
        self.dpctgan = PytorchDPSynthesizer(1.0, DPCTGAN(), None)

    def test_fit(self):
        self.dpctgan.fit(df, categorical_columns=['sex','educ','race','married'])
        assert self.dpctgan.gan.generator

    def test_sample(self):
        self.dpctgan.fit(df, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.dpctgan.sample(sample_size)
        assert synth_data.shape == df.shape

class TestPytorchDPSynthesizer_PATECTGAN:
    def setup(self):
        self.patectgan = PytorchDPSynthesizer(1.0, PATECTGAN(), None)

    def test_fit(self):
        self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
        assert self.patectgan.gan.generator

    def test_sample(self):
        self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.patectgan.sample(sample_size)
        assert synth_data.shape == df.shape

class TestPytorchDPSynthesizer_PATECTDRAGAN:
    def setup(self):
        self.patectgan = PytorchDPSynthesizer(1.0, PATECTGAN(regularization='dragan'), None)

    def test_fit(self):
        self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
        assert self.patectgan.gan.generator

    def test_sample(self):
        self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.patectgan.sample(sample_size)
        assert synth_data.shape == df.shape

# class TestPytorchDPSynthesizer_WPATECTDRAGAN:
#     def setup(self):
#         self.patectgan = PytorchDPSynthesizer(1.0, PATECTGAN(loss='wasserstein', regularization='dragan'), None)

#     def test_fit(self):
#         self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
#         assert self.patectgan.gan.generator

#     def test_sample(self):
#         self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
#         sample_size = len(df)
#         synth_data = self.patectgan.sample(sample_size)
#         assert synth_data.shape == df.shape
