import os
import subprocess

import pandas as pd
import pytest
from snsynth.transform.anonymization import AnonymizationTransformer

git_root_dir = (
    subprocess.check_output("git rev-parse --show-toplevel".split(" "))
    .decode("utf-8")
    .strip()
)
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))
narrow_df = pd.read_csv(
    csv_path, index_col=None, usecols=["age", "income", "married"], nrows=20
)

str_list = ["a", "b", "c"]
count = -1


def dummy_counter():
    global count
    count += 1
    return count


class TestAnonymization:
    # tests with Faker library
    def test_faker_builtin_with_kwargs(self):
        domain = "opendp.org"

        anon = AnonymizationTransformer("email", domain=domain)
        assert anon.fit_complete
        transformed = anon.transform(str_list)
        inversed = anon.inverse_transform(transformed)
        assert len(inversed) == len(str_list)
        assert all(email.split("@")[1] == domain for email in inversed)

    def test_faker_builtin_with_invalid_arguments(self):
        with pytest.raises(ValueError, match="arguments.*?invalid"):
            AnonymizationTransformer("email", obviously_invalid_kwarg="")

    def test_faker_invalid_builtin(self):
        with pytest.raises(ValueError, match="fake.*?not available"):
            AnonymizationTransformer("obviously_invalid_builtin")

    def test_faker_with_setup(self):
        anon = AnonymizationTransformer(
            "current_country", faker_setup={"locale": "de_DE"}
        )
        assert anon.fit_complete
        transformed = anon.transform(str_list)
        inversed = anon.inverse_transform(transformed)
        assert len(inversed) == len(str_list)
        assert all(country == "Germany" for country in inversed)

    def test_faker_with_invalid_setup(self):
        with pytest.raises(ValueError, match="faker_setup.*?invalid"):
            AnonymizationTransformer(
                "email", faker_setup={"providers": "obviously_invalid_provider"}
            )

    # tests with caller-provided functions
    def test_with_function(self):
        counter_sum = sum(range(1, len(str_list) + 1))

        anon = AnonymizationTransformer(dummy_counter)
        assert anon.fit_complete
        transformed = anon.transform(str_list)
        inversed = anon.inverse_transform(transformed)
        assert len(inversed) == len(str_list)
        assert sum(inversed) == counter_sum

    def test_with_parametrized_lambda(self):
        anon = AnonymizationTransformer(lambda x: x + 1, 1)
        assert anon.fit_complete
        transformed = anon.transform(str_list)
        inversed = anon.inverse_transform(transformed)
        assert len(inversed) == len(str_list)
        assert all(n == 2 for n in inversed)

    def _test_with_dummy_lambda(self, original, idx):
        anon = AnonymizationTransformer(lambda: "dummy")
        assert anon.fit_complete
        transformed = anon.transform(original, idx=idx)
        assert len(transformed) == len(original)
        assert all([len(a) == len(b) - 1 for a, b in zip(transformed, original)])
        inversed = anon.inverse_transform(transformed, idx=idx)
        assert len(inversed) == len(original)
        for row_inversed, row_original in zip(inversed, original):
            n_original = len(row_original)
            assert len(row_inversed) == n_original
            for i in range(n_original):
                if i == idx:
                    assert row_inversed[i] == "dummy"
                else:
                    assert row_inversed[i] == row_original[i]

    def test_one_column_with_idx(self):
        pums_tuples = [t[2:] for t in narrow_df.itertuples(index=False)]

        self._test_with_dummy_lambda(pums_tuples, 0)

    def test_two_columns_with_idx(self):
        pums_tuples = [t[1:] for t in narrow_df.itertuples(index=False)]

        self._test_with_dummy_lambda(pums_tuples, 0)
        self._test_with_dummy_lambda(pums_tuples, 1)

    def test_three_columns_with_idx(self):
        pums_tuples = [t for t in narrow_df.itertuples(index=False)]

        self._test_with_dummy_lambda(pums_tuples, 0)
        self._test_with_dummy_lambda(pums_tuples, 1)
        self._test_with_dummy_lambda(pums_tuples, 2)

    def test_fake_inbound(self):
        anon = AnonymizationTransformer(lambda: "dummy", fake_inbound=True)
        assert anon.fit_complete
        transformed = anon.transform(str_list)
        assert len(transformed) == len(str_list)
        assert all(t == "dummy" for t in transformed)
        inversed = anon.inverse_transform(transformed)
        assert len(inversed) == len(str_list)
        assert all(t == "dummy" for t in inversed)
