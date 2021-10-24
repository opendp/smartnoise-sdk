import os
import subprocess
import copy

import pandas as pd

from snsql.metadata import Metadata
from snsql.sql import PrivateReader
from snsql.sql.reader.pandas import PandasReader
from snsql.sql.parse import QueryParser

from snsql.sql.privacy import Privacy

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

schema = Metadata.from_file(meta_path)
df = pd.read_csv(csv_path)

#   Unit tests
#
class TestQuery:
    def test_count_exact(self):
        reader = PandasReader(df, schema)
        rs = reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS")
        assert(rs[1][0] == 1000)
    def test_empty_result(self):
        reader = PandasReader(df, schema)
        rs = reader.execute("SELECT age as a FROM PUMS.PUMS WHERE age > 100")
        assert(len(rs) == 1)
    def test_empty_result_typed(self):
        reader = PandasReader(df, schema)
        rs = reader.execute("SELECT age as a FROM PUMS.PUMS WHERE age > 100")
        trs = reader._to_df(rs)
        assert(len(trs) == 0)
    def test_group_by_exact_order(self):
        reader = PandasReader(df, schema)
        rs = reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs[1][0] == 451)
        assert(rs[2][0] == 549)
    def test_group_by_exact_order_desc(self):
        reader = PandasReader(df, schema)
        rs = reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs[1][0] == 549)
        assert(rs[2][0] == 451)
    def test_group_by_exact_order_expr_desc(self):
        reader = PandasReader(df, schema)
        rs = reader.execute("SELECT COUNT(*) * 5 AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs[1][0] == 549 * 5)
        assert(rs[2][0] == 451 * 5)
    # def test_group_by_noisy_order(self):
    #     reader = PandasReader(df, schema)
    #     private_reader = PrivateReader(schema, reader, 4.0)
    #     rs = private_reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
    #     assert(rs[1][0] < rs[2][0])
    # def test_group_by_noisy_order_desc(self):
    #     reader = PandasReader(df, schema)
    #     private_reader = PrivateReader(schema, reader, 4.0)
    #     rs = private_reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
    #     assert(rs[1][0] > rs[2][0])
    def test_group_by_noisy_typed_order(self):
        reader = PandasReader(df, schema)
        private_reader = PrivateReader(reader, schema, privacy=Privacy(epsilon=4.0))
        rs = private_reader.execute_df("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs['c'][0] < rs['c'][1])
    def test_group_by_noisy_typed_order_desc(self):
        reader = PandasReader(df, schema)
        private_reader = PrivateReader(reader, schema, privacy=Privacy(epsilon=4.0))
        rs = private_reader.execute_df("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs['c'][0] > rs['c'][1])
    def test_count_no_rows_exact_typed(self):
        reader = PandasReader(df, schema)
        query = QueryParser(schema).queries("SELECT COUNT(*) as c FROM PUMS.PUMS WHERE age > 100")[0]
        trs = reader._execute_ast_df(query)
        assert(trs['c'][0] == 0)
    def test_sum_no_rows_exact_typed(self):
        reader = PandasReader(df, schema)
        query = QueryParser(schema).queries("SELECT SUM(age) as c FROM PUMS.PUMS WHERE age > 100")[0]
        trs = reader._execute_ast_df(query)
        assert(trs['c'][0] == None)
    def test_sum_noisy(self):
        reader = PandasReader(df, schema)
        query = QueryParser(schema).queries("SELECT SUM(age) as age_total FROM PUMS.PUMS")[0]
        trs = reader._execute_ast_df(query)
        assert(trs['age_total'][0] > 1000)
    def test_sum_noisy_postprocess(self):
        reader = PandasReader(df, schema)
        private_reader = PrivateReader(reader, schema, privacy=Privacy(epsilon=1.0))
        trs = private_reader.execute_df("SELECT POWER(SUM(age), 2) as age_total FROM PUMS.PUMS")
        assert(trs['age_total'][0] > 1000 ** 2)
