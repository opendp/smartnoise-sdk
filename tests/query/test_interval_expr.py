import os
import subprocess

import pandas as pd
from pandasql import sqldf

from burdock.metadata import CollectionMetadata
from burdock.sql import PrivateReader, PandasReader
from burdock.sql.parse import QueryParser
from burdock.reader.rowset import TypedRowset


git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))
schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

#   Unit tests
#
class TestQuery:
    def test_group_by_noisy_typed_order_inter(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 1.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(rs['c'][0] < rs['c'][1])
        assert(len(rs.report['c'].intervals[0.95]) == 2)
        print(rs.report['c'].intervals[0.95])
        assert(all(ival.low < ival.high for ival in rs.report['c'].intervals[0.95]))
        assert(all(ival.low < ival.high for ival in rs.report['c'].intervals[0.985]))
        assert(all(outer.low < inner.low for inner, outer in zip(rs.report['c'].intervals[0.95], rs.report['c'].intervals[0.985])))
        assert(all(outer.high > inner.high for inner, outer in zip(rs.report['c'].intervals[0.95], rs.report['c'].intervals[0.985])))
    def test_group_by_noisy_typed_order_inter_constant(self):
        reader = PandasReader(schema, df)
        private_reader = PrivateReader(schema, reader, 1.0)
        rs = private_reader.execute_typed("SELECT COUNT(*) AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        rs2 = private_reader.execute_typed("SELECT COUNT(*) * 2 AS c, married AS m FROM PUMS.PUMS GROUP BY married ORDER BY c")
        assert(len(rs.report['c'].intervals[0.95]) == 2)
        assert(len(rs2.report['c'].intervals[0.95]) == 2)
        assert(all(a.low < b.low for a, b in zip(rs.report['c'].intervals[0.95], rs2.report['c'].intervals[0.95])))
        assert(all(a.low < b.low for a, b in zip(rs.report['c'].intervals[0.985], rs2.report['c'].intervals[0.985])))
