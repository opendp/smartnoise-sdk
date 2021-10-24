import os
import subprocess

import pandas as pd
import math
import pytest

from snsql.sql.dpsu import preprocess_df_from_query, run_dpsu
from snsql.metadata import Metadata
from snsql.sql import PrivateReader
from snsql import *
from snsql.sql.reader.pandas import PandasReader

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "reddit.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "reddit.csv"))


schema = Metadata.from_file(meta_path)
df = pd.read_csv(csv_path, index_col=0)



class TestDPSU:
    def test_preprocess_df_from_query(self):
        query = "SELECT ngram FROM reddit.reddit GROUP BY ngram"
        final_df = preprocess_df_from_query(schema, df, query)

        assert final_df is not None
        assert(len(final_df["group_cols"][0]) == 1)

    @pytest.mark.skip("strange error in CI")
    def test_run_dpsu(self):
        query = "SELECT ngram, COUNT(*) FROM reddit.reddit GROUP BY ngram"
        final_df = run_dpsu(schema, df, query, 3.0)

        assert final_df is not None
        assert len(final_df) > 0
        assert list(final_df) == list(df)
        assert not final_df.equals(df)
        assert len(final_df) < len(df)

    @pytest.mark.skip("max_ids needs to be overriden")
    def test_dpsu_vs_korolova(self):
        query = "SELECT ngram, COUNT(*) as n FROM reddit.reddit GROUP BY ngram ORDER BY n desc"
        reader = PandasReader(df, schema)
        private_reader = PrivateReader(reader, schema, privacy=Privacy(epsilon=3.0))
        private_reader.options.max_contrib = 10
        result = private_reader.execute_df(query)

        private_reader_korolova = PrivateReader(reader, schema, privacy=Privacy(epsilon=3.0))
        private_reader_korolova.options.dpsu = False
        private_reader_korolova.options.max_contrib = 10
        korolova_result = private_reader_korolova.execute_df(query)

        assert len(result['n']) > len(korolova_result['n'])
        assert len(final_df) < len(df)
