import os
import subprocess
import copy
import pytest

import pandas as pd
from pandasql import sqldf
import math

from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.sql import PrivateReader, PandasReader
from opendp.smartnoise.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

class TestQueryError:
    def test_err1(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].maxval = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, 4.0)
        with pytest.raises(ValueError):
            rs = private_reader.execute_df("SELECT SUM(income) FROM PUMS.PUMS")
    def test_err2(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].minval = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, 4.0)
        with pytest.raises(ValueError):
            rs = private_reader.execute_df("SELECT income, SUM(income) FROM PUMS.PUMS GROUP BY income")
    def test_ok1(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].maxval = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, 4.0)
        rs = private_reader.execute_df("SELECT income FROM PUMS.PUMS GROUP BY income")
    def test_ok2(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].maxval = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, 4.0)
        rs = private_reader.execute_df("SELECT COUNT(income) FROM PUMS.PUMS")
