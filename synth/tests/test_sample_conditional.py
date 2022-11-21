import os
import subprocess

import numpy as np
import pandas as pd
import pytest
from snsynth import Synthesizer

git_root_dir = (
    subprocess.check_output("git rev-parse --show-toplevel".split(" "))
    .decode("utf-8")
    .strip()
)
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))
narrrow_columns = ["income", "married"]
narrow_df = pd.read_csv(csv_path, index_col=None, usecols=narrrow_columns)


class TestSampleConditional:
    def test_n_row_invalid(self):
        dummy_synth = Synthesizer()
        for n_row in [-np.inf, -1, 0, 0.9]:
            with pytest.raises(ValueError, match="n_rows"):
                dummy_synth.sample_conditional(n_row, "dummy condition")

    def test_max_tries_invalid(self):
        dummy_synth = Synthesizer()
        for max_tries in [-np.inf, -1, 0, 0.9]:
            with pytest.raises(ValueError, match="max_tries"):
                dummy_synth.sample_conditional(
                    1, "dummy condition", max_tries=max_tries
                )

    def test_condition_unparsable(self):
        dummy_synth = Synthesizer()
        for condition in ["dummy condition", "age % 20", "WHERE age < 20"]:
            with pytest.raises(ValueError, match="parse.*?condition"):
                dummy_synth.sample_conditional(1, condition)

    def test_condition_invalid_column_name(self):
        dummy_synth = Synthesizer()

        # define small test data set
        columns = ["married"]
        data = [1, 0]
        invalid_condition = "age > 50"

        # test with DataFrame
        dummy_synth.sample = lambda _: pd.DataFrame(data=data, columns=columns)
        with pytest.raises(ValueError, match="evaluate.*?condition"):
            dummy_synth.sample_conditional(1, invalid_condition)

        # test with list of tuples
        dummy_synth.sample = lambda _: data
        with pytest.raises(ValueError, match="evaluate.*?condition"):
            dummy_synth.sample_conditional(1, invalid_condition, column_names=columns)

    def test_condition_no_column_names(self):
        dummy_synth = Synthesizer()
        dummy_synth.sample = lambda _: [55]
        with pytest.raises(ValueError, match="provide.*?column_names"):
            dummy_synth.sample_conditional(1, "age > 50")

    def test_max_tries_exceeded(self):
        dummy_synth = Synthesizer()
        dummy_synth.sample = lambda _: [1]

        samples = dummy_synth.sample_conditional(
            1, "married = 0", max_tries=5, column_names=["married"]
        )
        assert len(samples) == 0

    def test_with_data_frame(self):
        dummy_synth = Synthesizer()
        dummy_synth.sample = lambda _: narrow_df
        samples = dummy_synth.sample_conditional(10, "married = 0 AND income < 1000")

        assert len(samples) == 10
        assert isinstance(samples, pd.DataFrame)
        assert samples.dtypes.equals(narrow_df.dtypes)
        assert samples["income"].max() < 1000
        assert samples["married"].max() == 0

    def test_with_ndarray(self):
        dummy_synth = Synthesizer()
        dummy_synth.sample = lambda _: narrow_df.to_numpy()
        samples = dummy_synth.sample_conditional(
            10, "married = 0 AND income < 1000", column_names=narrrow_columns
        )

        assert len(samples) == 10
        assert isinstance(samples, np.ndarray)
        assert samples[0].max() < 1000
        assert samples[1].max() == 0

    def test_with_tuples(self):
        dummy_synth = Synthesizer()
        dummy_synth.sample = lambda _: list(narrow_df.itertuples(index=False))
        samples = dummy_synth.sample_conditional(
            10, "married = 0 AND income < 1000", column_names=narrrow_columns
        )

        assert len(samples) == 10
        assert isinstance(samples, list)
        income_max = max(s[0] for s in samples)
        assert income_max < 1000
        married_max = max(s[1] for s in samples)
        assert married_max == 0
