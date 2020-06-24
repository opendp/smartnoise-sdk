import os
import subprocess

import pandas as pd
from pandasql import sqldf
import math

from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.sql import PrivateReader, PandasReader
from opendp.whitenoise.sql.parse import QueryParser
from opendp.whitenoise.reader.rowset import TypedRowset

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

#   Unit tests
#
class TestQuery:
    def test_count_exact(self):
        reader = PandasReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS")
        assert(rs[1][0] == 1000)
    def test_empty_result(self):
        reader = PandasReader(schema, df)
        rs = reader.execute("SELECT age as a FROM PUMS.PUMS WHERE age > 100")
        assert(len(rs) == 1)
    def test_empty_result_typed(self):
        reader = PandasReader(schema, df)
        rs = reader.execute("SELECT age as a FROM PUMS.PUMS WHERE age > 100")
        trs = TypedRowset(rs, ['int'])
        assert(len(trs) == 0)
    def test_group_by_exact_order(self):
        reader = PandasReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs[1][0] == 451)
        assert(rs[2][0] == 549)
    def test_group_by_exact_order_desc(self):
        reader = PandasReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs[1][0] == 549)
        assert(rs[2][0] == 451)
    def test_group_by_exact_order_expr_desc(self):
        reader = PandasReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) * 5 AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs[1][0] == 549 * 5)
        assert(rs[2][0] == 451 * 5)
    def test_group_by_noisy_order(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 4.0)
        rs = private_reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs[1][0] < rs[2][0])
    # def test_group_by_noisy_order_desc(self):
    #     reader = PandasReader(schema, df)
    #     private_reader = PrivateReader(schema, reader, 4.0)
    #     rs = private_reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
    #     assert(rs[1][0] > rs[2][0])
    def test_group_by_noisy_typed_order(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 4.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs['c'][0] < rs['c'][1])
    def test_group_by_noisy_typed_order_desc(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 4.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs['c'][0] > rs['c'][1])
    def test_no_tau(self):
        # should never drop rows
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 4.0)
        for i in range(10):
            rs = private_reader.execute_typed("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 AND educ = '8'")
            assert(len(rs['c']) == 1)
    def test_no_tau_noisy(self):
        # should never drop rows
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 0.01)
        for i in range(10):
            rs = private_reader.execute_typed("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 AND educ = '8'")
            assert(len(rs['c']) == 1)
    def test_yes_tau(self):
        # should usually drop some rows
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 1.0, 1/10000)
        lengths = []
        for i in range(10):
            rs = private_reader.execute_typed("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 80 GROUP BY educ")
            lengths.append(len(rs['c']))
        l = lengths[0]
        assert(any([l != ll for ll in lengths]))
    def test_count_no_rows_exact_typed(self):
        reader = PandasReader(schema, df)
        query = QueryParser(schema).queries("SELECT COUNT(*) as c FROM PUMS.PUMS WHERE age > 100")[0]
        trs = reader.execute_ast_typed(query)
        assert(trs['c'][0] == 0)
    def test_sum_no_rows_exact_typed(self):
        reader = PandasReader(schema, df)
        query = QueryParser(schema).queries("SELECT SUM(age) as c FROM PUMS.PUMS WHERE age > 100")[0]
        trs = reader.execute_ast_typed(query)
        assert(trs['c'][0] == None)
    def test_empty_result_count_typed_notau_prepost(self):
        reader = PandasReader(schema, df)
        query = QueryParser(schema).queries("SELECT COUNT(*) as c FROM PUMS.PUMS WHERE age > 100")[0]
        private_reader = PrivateReader(schema, reader, 1.0)
        private_reader._execute_ast(query, True)
        for i in range(3):
            trs = private_reader._execute_ast(query, True)
            assert(len(trs) == 1)
    def test_sum_noisy(self):
        reader = PandasReader(schema, df)
        query = QueryParser(schema).queries("SELECT SUM(age) as age_total FROM PUMS.PUMS")[0]
        trs = reader.execute_ast_typed(query)
        assert(trs['age_total'][0] > 1000)
    def test_sum_noisy_postprocess(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 1.0)
        trs = private_reader.execute_typed("SELECT POWER(SUM(age), 2) as age_total FROM PUMS.PUMS")
        assert(trs['age_total'][0] > 1000 ** 2)
    def test_execute_with_dpsu(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 1.0)
        query = QueryParser(schema).queries("SELECT COUNT(*) AS c FROM PUMS.PUMS GROUP BY married")[0]
        assert(private_reader._get_reader(query) is not private_reader.reader)
    def test_execute_without_dpsu(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 1.0)
        query = QueryParser(schema).queries("SELECT COUNT(*) AS c FROM PUMS.PUMS GROUP BY married")[0]
        private_reader.options.use_dpsu = False
        assert(private_reader._get_reader(query) is private_reader.reader)
    def test_check_thresholds_gauss(self):
        # check tau for various privacy parameters
        epsilons = [0.1, 2.0]
        max_contribs = [1, 3]
        deltas = [10E-5, 10E-15]
        query = "SELECT COUNT(*) FROM PUMS.PUMS GROUP BY married"
        reader = PandasReader(schema, df)
        qp = QueryParser(schema)
        q = qp.query(query)        
        for eps in epsilons:
            for d in max_contribs:
                for delta in deltas:
                    # using slightly different formulations of same formula from different papers
                    # make sure private_reader round-trips
                    gaus_scale = math.sqrt(d) * math.sqrt(2 * math.log(1.25/delta))/eps
                    gaus_rho = 1 + gaus_scale * math.sqrt(2 * math.log(d / math.sqrt(2 * math.pi * delta)))
                    private_reader = PrivateReader(schema, reader, eps, delta)
                    q.max_ids = d # hijack the AST
                    r = private_reader.execute_ast(q)
                    assert(math.isclose(private_reader.tau, gaus_rho, rel_tol=0.03, abs_tol=2))