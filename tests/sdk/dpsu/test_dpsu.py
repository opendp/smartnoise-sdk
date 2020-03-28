import os
import subprocess

import pandas as pd
import math
import pytest

from opendp.whitenoise.sql.dpsu import preprocess_df_from_query, run_dpsu
from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

class TestDPSU:
    def test_preprocess_df_from_query(self):
        query = "SELECT married, educ FROM PUMS.PUMS GROUP BY married, educ"
        final_df = preprocess_df_from_query(schema, df, query)

        assert final_df is not None
        assert(len(final_df["group_cols"][0]) == 2)

    def test_dpsu_df(self):
        query = "SELECT married, educ FROM PUMS.PUMS GROUP BY married, educ"
        final_df = run_dpsu(schema, df, query, 3.0)
        print(final_df.columns)

        assert final_df is not None
        assert list(final_df) == list(df)
        assert not final_df.equals(df)
        assert len(final_df) < len(df)