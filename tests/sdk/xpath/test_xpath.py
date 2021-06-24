import os
import subprocess
import copy
import pytest

import pandas as pd
from pandasql import sqldf
import math

from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.sql import PrivateReader, PandasReader, SqlReader
from opendp.smartnoise.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))


from opendp.smartnoise.xpath.parse import XPath
p = XPath()

meta = CollectionMetadata.from_file(meta_path)
pums = pd.read_csv(csv_path)
query = 'SELECT AVG(age) + 3, STD(age), VAR(age), SUM(age) / 10, COUNT(age) + 2 FROM PUMS.PUMS'
q = QueryParser(meta).query(query)
reader = SqlReader.from_connection(pums, "pandas", metadata=meta)
priv = PrivateReader(reader, meta, 1.0)
subquery, root = priv.rewrite(query)

class TestXPathExecution:
    def test_all_root_descend(self):
        path = '//*' # returns value
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) > 40)
    def test_all_with_condition(self):
        path = '//*[@left]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 3)
    def test_root_predicate(self):
        path = '/Query[@select]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 1)
    def test_no_results_root_predicate(self):
        path = '/Query[@soos]' # returns []
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 0)
    def test_descend_attrib(self):
        path = '//@name'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 11)
    def test_child_all_nodes(self):
        path = '/Query/*'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 2)
    def test_child_all_attributes(self):
        path = '/Query/@*'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 7)
    def test_double_descend_to_att(self):
        path = '/Query/Select//ArithmeticExpression//AggFunction/@name'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 3)
    def test_simple_step(self):
        path = '/Query/Select'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 1)
    def test_simple_descend(self):
        path = '//Select'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 1)
