import subprocess
import os

import pytest
import string
import pandas as pd

from opendp.smartnoise.metadata import CollectionMetadata

try:
    from opendp.smartnoise.synthesizers.preprocessors import GeneralTransformer
    from opendp.smartnoise.synthesizers.pytorch import PytorchDPSynthesizer
    from opendp.smartnoise.synthesizers.pytorch.nn import PATEGAN
except:
    import logging
    test_logger = logging.getLogger(__name__)
    test_logger.warning("Requires torch and torchdp")


git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

@pytest.mark.torch
class TestDPGAN:
    def setup(self):
        self.pategan = PytorchDPSynthesizer(1.0, PATEGAN(1.0), GeneralTransformer())

    def test_fit(self):
        df_non_continuous = df[['sex','educ','race','married']]
        self.pategan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        assert self.pategan.gan.generator

    def test_sample(self):
        df_non_continuous = df[['sex','educ','race','married']]
        self.pategan.fit(df_non_continuous, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df_non_continuous)
        synth_data = self.pategan.sample(sample_size)
        assert synth_data.shape == df_non_continuous.shape
