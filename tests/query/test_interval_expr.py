import pytest
import pandas as pd

from burdock.query.sql.reader import DataFrameReader
from burdock.query.sql import MetadataLoader
from burdock.query.sql import QueryParser
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
    def test_group_by_noisy_typed_order_inter(self):
        reader = DataFrameReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 1.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs['c'][0] < rs['c'][1])
        assert(len(rs.intervals['c'][0]) == 2)
        assert(all(ival.low < ival.high for ival in rs.intervals['c'][0]))
        assert(all(ival.low < ival.high for ival in rs.intervals['c'][1]))
        assert(all(outer.low < inner.low for inner, outer in zip(rs.intervals['c'][0], rs.intervals['c'][1])))
        assert(all(outer.high > inner.high for inner, outer in zip(rs.intervals['c'][0], rs.intervals['c'][1])))
    def test_group_by_noisy_typed_order_inter_constant(self):
        reader = DataFrameReader(schema, df)
        private_reader = PrivateQuery(reader, schema, 1.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        rs2 = private_reader.execute_typed("SELECT COUNT(*) * 2 AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(len(rs.intervals['c'][0]) == 2)
        assert(len(rs2.intervals['c'][0]) == 2)
        assert(all(a.low < b.low for a, b in zip(rs.intervals['c'][0], rs2.intervals['c'][0])))
        assert(all(a.low < b.low for a, b in zip(rs.intervals['c'][1], rs2.intervals['c'][1])))
