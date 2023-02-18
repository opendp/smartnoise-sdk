import os
import subprocess
import pandas as pd
from sklearn import preprocessing
from snsynth import *

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

df = pd.read_csv(csv_path, index_col=None)

class TestFactory:
    def test_create_empty(self):
        for synth in Synthesizer.list_synthesizers():
            _ = Synthesizer.create(synth, epsilon=1.0)
    def test_fit_with_data_frame(self):
        # fit income by marital status
        narrow_df = df.drop(["age", "sex", "race", "educ"], axis=1)
        for synth in Synthesizer.list_synthesizers():
            synth = Synthesizer.create(synth, epsilon=2.0)
            print(f"Fitting {synth}...")
            synth.fit(narrow_df, preprocessor_eps=0.5)
            rows = synth.sample(100)
            assert (isinstance(rows, pd.DataFrame))
            assert (rows['income'].mean() > 1000 and rows['income'].mean() < 250000)
            assert (sum(rows['married'] == 1) > 1)
#             assert (sum(rows['married'] == 0) > 1)

