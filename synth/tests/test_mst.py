import subprocess
import os

import pytest
import numpy as np
import pandas as pd

from snsynth.mst import MSTSynthesizer

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

df = pd.read_csv(csv_path, index_col=0)
df = df.drop(["income"], axis=1)
df = df[df > 0].dropna()
df = df.sample(frac=0.05, random_state=42)

class TestMST:

    Domains = {
        "test": "synth/tests/test-domain.json"
    }

    def setup(self):
        self.mst = MSTSynthesizer(domains_dict=self.Domains)

    def test_fit(self):
        self.df_non_continuous = df[['sex','educ','race','married']]
        self.mst.fit(self.df_non_continuous)
        assert self.mst

    def test_sample(self):
        self.df_non_continuous = df[['sex','educ','race','married']]
        self.mst.fit(self.df_non_continuous)
        sample_size = len(df)
        synth_data = self.mst.sample(sample_size)
        assert synth_data.shape == self.df_non_continuous.shape