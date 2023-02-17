import pandas as pd

from unittest import TestCase
from snsynth.aim import AIMSynthesizer


class TestAIM(TestCase):
    input_data_path = '../datasets/example.csv'
    aim = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.example_df = pd.read_csv(cls.input_data_path)
        cls.aim = AIMSynthesizer()

    def test_fit(self):
        self.aim.fit(self.example_df)
        assert self.aim

    def test_sample(self):
        self.aim.fit(self.example_df)
        sample_size = len(self.example_df)
        synth_data = self.aim.sample(sample_size)
        assert synth_data.shape == self.example_df.shape
