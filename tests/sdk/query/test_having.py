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
from opendp.smartnoise.reader.rowset import TypedRowset

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))


class TestBaseTypes:
    def setup_class(cls):
        meta = CollectionMetadata.from_file(meta_path)
        meta["PUMS.PUMS"].censor_dims = False
        df = pd.read_csv(csv_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, 10.0, 10E-3)
        cls.reader = private_reader

    def test_queries(self):
        query = "SELECT age, sex, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, sex HAVING income > 100000"
        res = self.reader.execute(query)
        assert len(res) < 100 # actual is 14, but noise is huge

        query = "SELECT age, sex, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, sex HAVING sex = 1"
        res = self.reader.execute(query)
        assert len(res) == 74

        query = "SELECT age, sex, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, sex HAVING income > 100000 OR sex = 1"
        res = self.reader.execute(query)
        assert len(res) > 80 and len(res) < 150

        query = "SELECT age, COUNT(*) FROM PUMS.PUMS GROUP BY age HAVING age < 30 OR age > 60"
        res = self.reader.execute(query)
        assert len(res) == 43

        # this one is indeterminate behavior based on engine, but works on PrivateReader
        query = "SELECT age * 1000 as age, COUNT(*) FROM PUMS.PUMS GROUP BY age HAVING age < 30000 OR age > 60000"
        res = self.reader.execute(query)
        assert len(res) == 43

        query = "SELECT age as age, COUNT(*) FROM PUMS.PUMS GROUP BY age HAVING age * 1000 < 30000 OR age * 2 > 120"
        res = self.reader.execute(query)
        assert len(res) == 43

        query = "SELECT age, COUNT(*) AS n FROM PUMS.PUMS GROUP BY age HAVING (age < 30 OR age > 60) AND n > 10"
        res = self.reader.execute(query)
        assert len(res) < 25 # [len is 16 for non-private]

        query = "SELECT age, COUNT(*) * 1000 AS n FROM PUMS.PUMS GROUP BY age HAVING (age < 30 OR age > 60) AND n > 10000"
        res = self.reader.execute(query)
        assert len(res) < 25  #[len is 16 for non-private]

        query = "SELECT age, COUNT(*) AS n FROM PUMS.PUMS GROUP BY age HAVING (age < 30 OR age > 60) AND n * 100 / 2 > 500"
        res = self.reader.execute(query)
        assert len(res) < 25  #[len is 16 for non-private]

class TestOtherTypes:
    def setup_class(self):
        meta = CollectionMetadata.from_file(meta_path)
        meta["PUMS.PUMS"].censor_dims = False
        meta["PUMS.PUMS"]["sex"].type = "int"
        meta["PUMS.PUMS"]["educ"].type = "int"
        meta["PUMS.PUMS"]["married"].type = "bool"
        df = pd.read_csv(csv_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, 10.0, 10E-3)
        self.reader = private_reader

    def test_queries(self):
        query = "SELECT age, sex, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, sex HAVING income > 100000"
        res = self.reader.execute(query)
        assert len(res) < 85 # actual is 14, but noise is huge

        query = "SELECT age, sex, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, sex HAVING sex = 1"
        res = self.reader.execute(query)
        assert len(res) == 74

        query = "SELECT age, sex, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, sex HAVING income > 100000 OR sex = 1"
        res = self.reader.execute(query)
        assert len(res) > 80 and len(res) < 150

        query = "SELECT age, COUNT(*) FROM PUMS.PUMS GROUP BY age HAVING age < 30 OR age > 60"
        res = self.reader.execute(query)
        assert len(res) == 43

        # this one is indeterminate behavior based on engine, works with PrivateReader
        query = "SELECT age * 1000 as age, COUNT(*) FROM PUMS.PUMS GROUP BY age HAVING age < 30000 OR age > 60000"
        res = self.reader.execute(query)
        assert len(res) == 43

        query = "SELECT age as age, COUNT(*) FROM PUMS.PUMS GROUP BY age HAVING age * 1000 < 30000 OR age * 2 > 120"
        res = self.reader.execute(query)
        assert len(res) == 43

        query = "SELECT age, COUNT(*) AS n FROM PUMS.PUMS GROUP BY age HAVING (age < 30 OR age > 60) AND n > 10"
        res = self.reader.execute(query)
        assert len(res) < 25  #[len is 16 for non-private]

        query = "SELECT age, COUNT(*) * 1000 AS n FROM PUMS.PUMS GROUP BY age HAVING (age < 30 OR age > 60) AND n > 10000"
        res = self.reader.execute(query)
        assert len(res) < 25  #[len is 16 for non-private]

        query = "SELECT age, COUNT(*) AS n FROM PUMS.PUMS GROUP BY age HAVING (age < 30 OR age > 60) AND n * 100 / 2 > 500"
        res = self.reader.execute(query)
        assert len(res) < 25 # [len is 16 for non-private]

        query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING married = 1"
        res = self.reader.execute(query)
        assert len(res) == 72

        query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING n > 10 OR married = 1"
        res = self.reader.execute(query)
        assert len(res) > 75

        #query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING married"
        #res = self.reader.execute(query)
        #print(len(res))
        #assert len(res) == 72

        #query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING n > 10 OR married"
        #res = self.reader.execute(query)
        #print(len(res))
        #assert len(res) > 75

        query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING married = 0"
        res = self.reader.execute(query)
        assert len(res) == 73

        query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING n > 10 OR married = 0"
        res = self.reader.execute(query)
        assert len(res) > 75

        #query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING NOT married"
        #res = self.reader.execute(query)
        #print(len(res))
        #assert len(res) == 73

        query = "SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married HAVING n > 10 OR NOT married"
        res = self.reader.execute(query)
        assert len(res) > 75
