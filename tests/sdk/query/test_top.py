import os
import subprocess
import copy
from opendp.smartnoise.sql.privacy import Privacy
import pytest

import pandas as pd
from pandasql import sqldf
import math

from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.sql import PrivateReader, PandasReader
from opendp.smartnoise.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

csv_path = '/Users/joshuaallen/Source/dp-test-datasets/data/PUMS_california_demographics_1000/data.csv'

pums_schema_path = os.path.join("datasets", "PUMS.yaml")


class TestTopAndLimit:
    def setup_class(cls):
        meta = CollectionMetadata.from_file(meta_path)
        meta["PUMS.PUMS"].censor_dims = False
        df = pd.read_csv(csv_path)
        print(df)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, 10.0, 0.1)
        cls.reader = private_reader

    def test_queries(self, test_databases):
        query = 'SELECT TOP 20 age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married ORDER BY married, age DESC'
        #query = 'SELECT COUNT(*) AS n FROM PUMS.PUMS GROUP BY race'
        privacy = Privacy(10.0, 0.1)
        tdb = test_databases
        print(tdb.engines['pandas'].connections['PUMS_pid'])
        readers = tdb.create_private_readers(privacy=privacy, database='PUMS_pid')

        for reader in readers:
            print(reader.reader)
            res = reader.execute_df(query)
            print(res)
            res = reader.reader.execute('SELECT COUNT(*)')
            print(res)
            #assert len(res) == 21

        reader = self.reader
        print("~~~~~")
        print(reader.reader)
        res = reader.execute(query)
        print(res)
        assert len(res) == 21

        query = 'SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married ORDER BY married, age DESC LIMIT 10'
        res = reader.execute(query)
        assert len(res) == 11
        # run the same query with exact reader. Since ORDER BY is
        # on non-private dimension, order will be the same
        res_e = reader.reader.execute(query)
        assert len(res_e) == 11
        ages = [r[0] for r in res[1:]]
        ages_e = [r[0] for r in res_e[1:]]
        assert all([age == age_e for (age, age_e) in zip(ages, ages_e)])

        query = 'SELECT age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married ORDER BY income DESC LIMIT 50'
        res = reader.execute(query)
        assert len(res) == 51
        # run the same query with exact reader. Since ORDER BY is
        # on non-private dimension, order will be different
        res_e = reader.reader.execute(query)
        assert len(res_e) == 51
        ages = [r[0] for r in res[1:]]
        ages_e = [r[0] for r in res_e[1:]]
        assert not all([age == age_e for (age, age_e) in zip(ages, ages_e)])
