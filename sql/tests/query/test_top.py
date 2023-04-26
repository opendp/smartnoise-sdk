import os
import subprocess
from snsql.sql.privacy import Privacy

import pandas as pd

from snsql.metadata import Metadata
from snsql.sql import PrivateReader
from snsql.sql.reader.pandas import PandasReader

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

pums_schema_path = os.path.join("datasets", "PUMS.yaml")


class TestTopAndLimit:
    def setup_class(cls):
        meta = Metadata.from_file(meta_path)
        meta["PUMS.PUMS"].censor_dims = False
        df = pd.read_csv(csv_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, privacy=Privacy(epsilon=10.0, delta=0.1))
        cls.reader = private_reader
    def test_order_limit(self, test_databases):
        """
        The top 3 education levels are each more than double the 4th,
        so a SELECT TOP will reliably give the top 3
        """
        privacy = Privacy(epsilon=3.0, delta=1/1000)
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            res = reader.execute('SELECT TOP 5 educ, COUNT(*) AS n FROM PUMS.PUMS GROUP BY educ ORDER BY n DESC')
            res = test_databases.to_tuples(res)
            educs = [str(r[0]) for r in res]
            assert('9' in educs)
            assert('13' in educs)
            assert('11' in educs)
    def test_no_order_limit(self, test_databases):
        """
        The top 3 education levels are each more than double the 4th,
        so a SELECT TOP will reliably give the top 3
        """
        privacy = Privacy(epsilon=3.0, delta=1/1000)
        readers = test_databases.get_private_readers(database='PUMS_dup', privacy=privacy)
        for reader in readers:
            res = reader.execute('SELECT educ, COUNT(*) AS n FROM PUMS.PUMS GROUP BY educ LIMIT 4')
            res = test_databases.to_tuples(res)
            educs = [str(r[0]) for r in res]
            top_educs = ['9', '13', '11', '12']
            assert(not all([a == b for a, b in zip(educs, top_educs)]))
    def test_queries(self, test_databases):
        query = 'SELECT TOP 20 age, married, COUNT(*) AS n, SUM(income) AS income FROM PUMS.PUMS GROUP BY age, married ORDER BY married, age DESC'
        privacy = Privacy(10.0, 0.1)
        tdb = test_databases
        readers = tdb.get_private_readers(privacy=privacy, database='PUMS_pid', overrides={'censor_dims': False})

        for reader in readers:
            if reader.engine == "spark":
                continue
            res = test_databases.to_tuples(reader.execute(query))
            assert len(res) == 21

        reader = self.reader
        res = reader.execute(query)
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
