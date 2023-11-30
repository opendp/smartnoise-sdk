import os
import subprocess
import copy
import pytest

import pandas as pd
import math

from snsql.metadata import Metadata
from snsql.sql import PrivateReader
from snsql.sql._mechanisms.base import Mechanism
from snsql.sql.reader.pandas import PandasReader
from snsql.sql.parse import QueryParser

from snsql.sql.privacy import Privacy, Stat

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

schema = Metadata.from_file(meta_path)
df = pd.read_csv(csv_path)

#   Unit tests
#
class TestQueryThresholds:
    def test_yes_tau_laplace(self, test_databases):
        # should drop approximately half of educ bins
        privacy = Privacy(epsilon=1.0, delta=1/1000)
        privacy.mechanisms.map[Stat.threshold] = Mechanism.laplace
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            rs = test_databases.to_tuples(reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 70 GROUP BY educ"))
            assert(len(rs) >= 2 and len(rs) <= 8)
    def test_yes_tau_laplace_row(self, test_databases):
        # should drop approximately half of educ bins
        privacy = Privacy(epsilon=1.0, delta=1/1000)
        privacy.mechanisms.map[Stat.threshold] = Mechanism.laplace
        readers = test_databases.get_private_readers(database='PUMS', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            rs = test_databases.to_tuples(reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 70 GROUP BY educ"))
            assert(len(rs) >= 2 and len(rs) <= 8)
    def test_yes_tau_gauss(self, test_databases):
        # should drop approximately half of educ bins
        privacy = Privacy(epsilon=1.0, delta=1/1000)
        privacy.mechanisms.map[Stat.threshold] = Mechanism.discrete_laplace
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            rs = test_databases.to_tuples(reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 70 GROUP BY educ"))
            assert(len(rs) >= 2 and len(rs) <= 12)
    def test_yes_tau_gauss_row(self, test_databases):
        # should drop approximately half of educ bins
        privacy = Privacy(epsilon=1.0, delta=1/1000)
        privacy.mechanisms.map[Stat.threshold] = Mechanism.discrete_laplace
        readers = test_databases.get_private_readers(database='PUMS', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            rs = test_databases.to_tuples(reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 70 GROUP BY educ"))
            assert(len(rs) >= 2 and len(rs) <= 8)

    def test_yes_tau_laplace_no_group(self, test_databases):
        # This should always return empty, because it pinpoints a small cohort
        privacy = Privacy(epsilon=1.0, delta=1/100_000)
        privacy.mechanisms.map[Stat.threshold] = Mechanism.laplace
        readers = test_databases.get_private_readers(database='PUMS', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            rs = test_databases.to_tuples(reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 70 AND educ < 2"))
            assert(len(rs) <= 1)
    @pytest.mark.skip("strange error in CI")
    def test_execute_with_dpsu(self):
        schema_dpsu = copy.copy(schema)
        schema_dpsu["PUMS.PUMS"].use_dpsu = True
        reader = PandasReader(df, schema_dpsu)
        private_reader = PrivateReader(reader, schema_dpsu, 1.0)
        assert(private_reader._options.use_dpsu == True)
        query = QueryParser(schema_dpsu).queries("SELECT COUNT(*) AS c FROM PUMS.PUMS GROUP BY married")[0]
        assert(private_reader._get_reader(query) is not private_reader.reader)
    def test_execute_without_dpsu(self):
        schema_no_dpsu = copy.copy(schema)
        schema_no_dpsu["PUMS.PUMS"].use_dpsu = False
        reader = PandasReader(df, schema_no_dpsu)
        private_reader = PrivateReader(reader, schema_no_dpsu, privacy=Privacy(epsilon=1.0))
        assert(private_reader._options.use_dpsu == False)
        query = QueryParser(schema_no_dpsu).queries("SELECT COUNT(*) AS c FROM PUMS.PUMS GROUP BY married")[0]
        assert(private_reader._get_reader(query) is private_reader.reader)
    def test_check_thresholds_gauss(self):
        # check tau for various privacy parameters
        epsilons = [0.1, 2.0]
        max_contribs = [1, 3]
        deltas = [10E-5, 10E-15]
        query = "SELECT COUNT(*) FROM PUMS.PUMS GROUP BY married"
        reader = PandasReader(df, schema)
        for eps in epsilons:
            for d in max_contribs:
                for delta in deltas:
                    privacy = Privacy(epsilon=eps, delta=delta)
                    privacy.mechanisms.map[Stat.threshold] = Mechanism.gaussian
                    # crude smoke test; compare threshold from discrete gaussian to threshold from gaussian,
                    # and check that the threshold isn't widly different
                    gaus_scale = math.sqrt(d) * math.sqrt(2 * math.log(1.25/delta))/eps
                    gaus_rho = 1 + gaus_scale * math.sqrt(2 * math.log(d / math.sqrt(2 * math.pi * delta)))
                    schema_c = copy.copy(schema)
                    schema_c["PUMS.PUMS"].max_ids = d
                    qp = QueryParser(schema_c)
                    q = qp.query(query)
                    private_reader = PrivateReader(reader, metadata=schema_c, privacy=privacy)
                    r = private_reader._execute_ast(q)
                    assert(private_reader.tau < gaus_rho * 3 and private_reader.tau > gaus_rho / 3)
    def test_empty_result_count_typed_notau_prepost(self):
        schema_all = copy.deepcopy(schema)
        schema_all['PUMS.PUMS'].censor_dims = False
        reader = PandasReader(df, schema)
        query = QueryParser(schema).queries("SELECT COUNT(*) as c FROM PUMS.PUMS WHERE age > 100")[0]
        private_reader = PrivateReader(reader, schema_all, privacy=Privacy(epsilon=1.0))
        private_reader._execute_ast(query, True)
        for i in range(3):
            trs = private_reader._execute_ast(query, True)
            assert(len(trs) == 1)
    def test_no_tau(self, test_databases):
        # should never drop rows
        privacy = Privacy(epsilon=4.0)
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy, overrides={'censor_dims': False})
        assert(len(readers) > 0)
        for reader in readers:
            if reader.engine == "spark":
                continue
            for _ in range(10):
                rs = reader.execute_df("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 AND educ = '8'")
                assert(len(rs['c']) == 1)
    def test_no_tau_noisy(self, test_databases):
        # should never drop rows
        privacy = Privacy(epsilon=4.0)
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy, overrides={'censor_dims': False})
        assert(len(readers) > 0)
        for reader in readers:
            if reader.engine == "spark":
                continue
            for i in range(10):
                reader._options.censor_dims = False
                rs = reader.execute_df("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 AND educ = '8'")
                assert(len(rs['c']) == 1)
