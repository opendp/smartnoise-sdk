import pytest
import pandas as pd

from burdock.query.sql.reader import CSVReader
from burdock.query.sql import MetadataLoader
from burdock.query.sql.private.query import PrivateQuery
from burdock.query.sql.reader.rowset import TypedRowset
from pandasql import sqldf

meta_path = "service/datasets/PUMS.yaml"
csv_path = "service/datasets/PUMS.csv"
schema = MetadataLoader(meta_path).read_schema()
df = pd.read_csv(csv_path)

#   Unit tests
#
class TestQuery:
    def test_count_exact(self):
        reader = CSVReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) AS c FROM PUMS.PUMS")
        assert(rs[1][0] == 1000)
    def test_group_by_exact_order(self):
        reader = CSVReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs[1][0] == 451)
        assert(rs[2][0] == 549)
    def test_group_by_exact_order_desc(self):
        reader = CSVReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs[1][0] == 549)
        assert(rs[2][0] == 451)
    def test_group_by_exact_order_expr_desc(self):
        reader = CSVReader(schema, df)
        rs = reader.execute("SELECT COUNT(*) * 5 AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs[1][0] == 549 * 5)
        assert(rs[2][0] == 451 * 5)
    def test_group_by_noisy_order(self):
        reader = CSVReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 1.0)
        rs = private_reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs[1][0] < rs[2][0])
    def test_group_by_noisy_order_desc(self):
        reader = CSVReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 1.0)
        rs = private_reader.execute("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs[1][0] > rs[2][0])
    def test_group_by_noisy_typed_order(self):
        reader = CSVReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 1.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs['c'][0] < rs['c'][1])
    def test_group_by_noisy_typed_order_desc(self):
        reader = CSVReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 1.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c DESC")
        assert(rs['c'][0] > rs['c'][1])
    def test_no_tau(self):
        # should never drop rows
        reader = CSVReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 4.0)
        for i in range(10):
            rs = private_reader.execute_typed("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 AND educ = '8'")
            assert(len(rs['c']) == 1)
    def test_no_tau_noisy(self):
        # should never drop rows
        reader = CSVReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 0.01)
        for i in range(10):
            rs = private_reader.execute_typed("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 AND educ = '8'")
            assert(len(rs['c']) == 1)
    def test_yes_tau(self):
        # should usually drop some rows
        reader = CSVReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 0.01)
        lengths = []
        for i in range(10):
            rs = private_reader.execute_typed("SELECT COUNT(*) AS c FROM PUMS.PUMS WHERE age > 90 GROUP BY educ")
            lengths.append(len(rs['c']))
        l = lengths[0]
        print(lengths)
        assert(any([l != ll for ll in lengths]))
