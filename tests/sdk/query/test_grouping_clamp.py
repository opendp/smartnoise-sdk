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

class TestGroupingClamp:
    def test_clamp_on(self):
        df = pd.read_csv(csv_path)
        meta = CollectionMetadata.from_file(meta_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, 30.0) # want small error
        meta["PUMS.PUMS"]["income"].maxval = 100

        query = "SELECT AVG(income) AS income FROM PUMS.PUMS"
        res = private_reader.execute(query)
        assert(res[1][0] < 150.0)
    def test_clamp_off(self):
        df = pd.read_csv(csv_path)
        meta = CollectionMetadata.from_file(meta_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, 30.0) # want small error
        meta["PUMS.PUMS"]["income"].maxval = 100

        query = "SELECT income, COUNT(pid) AS n FROM PUMS.PUMS GROUP BY income"
        res = private_reader.execute(query)
        assert(len(res) > 40)
