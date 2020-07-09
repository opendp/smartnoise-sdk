import subprocess
import os

import pytest
import string
import pandas as pd

from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
from opendp.whitenoise.synthesizers.pytorch.nn import DPGAN
from opendp.whitenoise.synthesizers.pytorch.pytorch_synthesizer import PytorchDPSynthesizer

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

dpgan = PytorchDPSynthesizer(GeneralTransformer(), DPGAN())

@pytest.mark.torch
class TestPytorchDPSynthesizer:
    @pytest.mark.torch
    def test_fit(self):
        dpgan.fit(df)
        assert dpgan.gan.generator
    
    @pytest.mark.torch
    def test_sample(self):
        sample_size = len(df)
        synth_data = dpgan.sample(sample_size)
        assert synth_data.shape == df.shape