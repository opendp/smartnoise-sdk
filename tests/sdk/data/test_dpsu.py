import os
import subprocess

import pandas as pd
import math
import pytest

from opendp.whitenoise.data.dpsu import preprocess_df_from_query, policy_laplace, dpsu_df
from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.sql import PrivateReader, PandasReader
from opendp.whitenoise.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

class TestDPSU:
    def test_preprocess_df_from_query(self):
        query = "SELECT married, educ FROM PUMS.PUMS GROUP BY married, educ"
        df = preprocess_df_from_query(meta_path, csv_path, query, "PUMS.PUMS")

        assert df is not None
        assert(len(df["group_cols"][0]) == 2)

    def test_dpsu_df(self):
        original_df = pd.read_csv(csv_path)
        query = "SELECT married, educ FROM PUMS.PUMS GROUP BY married, educ"
        df = preprocess_df_from_query(meta_path, csv_path, query, "PUMS.PUMS")
        final_df = dpsu_df(original_df, policy_laplace(df, 30, 3, math.exp(-10)))

        assert final_df is not None
        assert not final_df.equals(original_df)
        assert len(final_df) < len(original_df)