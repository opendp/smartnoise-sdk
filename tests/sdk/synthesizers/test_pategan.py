import subprocess
import os

import pytest
import string
import pandas as pd

from opendp.whitenoise.metadata import CollectionMetadata

try:
    from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
    from opendp.whitenoise.synthesizers.pytorch.pytorch_synthesizer import PytorchDPSynthesizer
    from opendp.whitenoise.synthesizers.pytorch.nn import PATEGAN
except:
    import logging
    test_logger = logging.getLogger(__name__)
    test_logger.warning("Requires torch and torchdp")


git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

@pytest.mark.torch
class TestDPGAN:
    def setup(self):
        self.pategan = PytorchSynthesizer(PATEGAN(), GeneralTransformer())

    def test_fit(self):
        self.pategan.fit(df)
        assert self.pategan.gan.generator
    
    def test_sample(self):
        self.pategan.fit(df)
        sample_size = len(df)
        synth_data = self.pategan.sample(sample_size)
        assert synth_data.shape == df.shape