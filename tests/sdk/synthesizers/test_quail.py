import subprocess
import os

import pytest
import string
import pandas as pd

from opendp.smartnoise.metadata import CollectionMetadata
from diffprivlib.models import LogisticRegression as DPLR

try:
    from opendp.smartnoise.synthesizers.preprocessors import GeneralTransformer
    from opendp.smartnoise.synthesizers.pytorch import PytorchDPSynthesizer
    from opendp.smartnoise.synthesizers.pytorch.nn import PATECTGAN
    from opendp.smartnoise.synthesizers import QUAILSynthesizer

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
class TestQUAIL:
    def setup(self):
        def QuailClassifier(epsilon):
            return DPLR(epsilon)

        def QuailSynth(epsilon):
            return PytorchDPSynthesizer(preprocessor=None,
                            gan=PATECTGAN(epsilon, loss='cross_entropy', batch_size=50, pack=1, sigma=5.0))

        self.quail = QUAILSynthesizer(3.0, QuailSynth, QuailClassifier, 'married')

    def test_fit(self):
        self.quail.fit(df, categorical_columns=['sex','educ','race','married'])
        assert self.quail.private_synth

    def test_sample(self):
        self.quail.fit(df, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.quail.sample(sample_size)
        assert synth_data.shape == df.shape
