import subprocess
import os

import pytest
import string
import pandas as pd

from diffprivlib.models import LogisticRegression as DPLR

try:
    from snsynth.preprocessors import GeneralTransformer
    from snsynth.pytorch import PytorchDPSynthesizer
    from snsynth.pytorch.nn import PATECTGAN
    from snsynth import QUAILSynthesizer

except:
    import logging
    test_logger = logging.getLogger(__name__)
    test_logger.warning("Requires torch and torchdp")


git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

df = pd.read_csv(csv_path)
del df['income']

@pytest.mark.torch
class TestQUAIL:
    def setup(self):
        def QuailClassifier(epsilon):
            return DPLR(epsilon=epsilon)

        def QuailSynth(epsilon):
            return PytorchDPSynthesizer(epsilon=epsilon, preprocessor=None,
                            gan=PATECTGAN(loss='cross_entropy', batch_size=50, pac=1))

        self.quail = QUAILSynthesizer(3.0, QuailSynth, QuailClassifier, 'married', eps_split=0.8)

    def test_fit(self):
        categorical_columns = [col for col in df.columns if col != 'married']
        self.quail.fit(df, categorical_columns=categorical_columns)
        assert self.quail.private_synth

    def test_sample(self):
        categorical_columns = [col for col in df.columns if col != 'married']
        self.quail.fit(df, categorical_columns=categorical_columns)
        sample_size = len(df)
        synth_data = self.quail.sample(sample_size)
        assert synth_data.shape == df.shape
