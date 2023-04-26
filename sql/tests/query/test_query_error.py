import os
import subprocess
import copy
import pytest

import pandas as pd

from snsql.metadata import Metadata
from snsql.sql import PrivateReader
from snsql.sql.reader.pandas import PandasReader
from snsql import *

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

schema = Metadata.from_file(meta_path)
df = pd.read_csv(csv_path)

class TestQueryError:
    def test_err1(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].upper = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, privacy=Privacy(epsilon=4.0))
        with pytest.raises(ValueError):
            rs = private_reader.execute_df("SELECT SUM(income) FROM PUMS.PUMS")
    def test_err2(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].lower = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, privacy=Privacy(epsilon=4.0))
        with pytest.raises(ValueError):
            rs = private_reader.execute_df("SELECT income, SUM(income) FROM PUMS.PUMS GROUP BY income")
    def test_ok1(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].upper = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, privacy=Privacy(epsilon=4.0))
        rs = private_reader.execute_df("SELECT income FROM PUMS.PUMS GROUP BY income")
    def test_ok2(self):
        s = copy.copy(schema)
        s["PUMS.PUMS"]["income"].upper = None
        reader = PandasReader(df, s)
        private_reader = PrivateReader(reader, s, privacy=Privacy(epsilon=4.0))
        rs = private_reader.execute_df("SELECT COUNT(income) FROM PUMS.PUMS")
