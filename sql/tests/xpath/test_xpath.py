import os
import subprocess

import pandas as pd

from snsql.metadata import Metadata
from snsql.sql import PrivateReader
from snsql.sql.reader.base import SqlReader
from snsql.sql.parse import QueryParser
from snsql import *

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))


from snsql.xpath.parse import XPath
p = XPath()

meta = Metadata.from_file(meta_path)
pums = pd.read_csv(csv_path)
query = 'SELECT AVG(age) + 3, STD(age), VAR(age), SUM(age) / 10, COUNT(age) + 2 FROM PUMS.PUMS'
q = QueryParser(meta).query(query)
reader = SqlReader.from_connection(pums, "pandas", metadata=meta)
priv = PrivateReader(reader, meta, privacy=Privacy(epsilon=1.0))
subquery, root = priv._rewrite(query)

class TestXPathExecutionNoRewrite:
    def test_all_root_descend(self):
        path = '//*' # returns value
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) > 40)
        assert(str(xx) == path)
    def test_all_with_condition(self):
        path = '//*[@left]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 3)
        assert(str(xx) == path)
    def test_root_predicate(self):
        path = '/Query[@select]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 1)
        assert(str(xx) == path)
    def test_no_results_root_predicate(self):
        path = '/Query[@soos]' # returns []
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 0)
        assert(str(xx) == path)
    def test_descend_attrib(self):
        path = '//@name'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 11)
        assert(str(xx) == path)
    def test_child_all_nodes(self):
        path = '/Query/*'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 2)
        assert(str(xx) == path)
    def test_child_all_attributes(self):
        path = '/Query/@*'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 7)
        assert(str(xx) == path)
    def test_double_descend_to_att(self):
        path = '/Query/Select//ArithmeticExpression//AggFunction/@name'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) >= 3)
        assert(str(xx) == path)
    def test_simple_step(self):
        path = '/Query/Select'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 1)
        assert(str(xx) == path)
    def test_simple_descend(self):
        path = '//Select'
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 1)
        assert(str(xx) == path)
    def test_attribute_to_literal(self):
        path = "//ArithmeticExpression[@right > 2]"
        xx = p.parse(path)
        res = xx.evaluate(q, 0)
        assert(len(res) == 2)
        assert(str(xx) == path)
    def test_literal_to_attribute(self):
        path = "//ArithmeticExpression[2.01 < @right]"
        xx = p.parse(path)
        res = xx.evaluate(q)
        assert(len(res) == 2)
        assert(str(xx) == path)

class TestXPathExecutionWithRewrite:
    def test_all_root_descend(self):
        path = '//*' # returns value
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) > 230)
    def test_all_with_condition(self):
        path = '//*[@left]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) >= 17)
    def test_root_predicate(self):
        path = '/Query[@select]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) == 1)
    def test_no_results_root_predicate(self):
        path = '/Query[@soos]' # returns []
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) == 0)
    def test_descend_attrib(self):
        path = '//@name'
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) >= 41)
    def test_child_all_nodes(self):
        path = '/Query/*'
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) >= 2)
    def test_child_all_attributes(self):
        path = '/Query/@*'
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) >= 7)
    def test_double_descend_to_att(self):
        path = '/Query/Select//ArithmeticExpression//AggFunction/@name'
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) >= 0)  # AggFunctions rewritten
    def test_simple_step(self):
        path = '/Query/Select'
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) == 1)
    def test_simple_descend(self):
        path = '//Select'
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) == 4) # nested queries
    def test_string_equal(self):
        path = "//AggFunction[@name == 'COUNT']"
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) == 2)
        assert(str(xx) == path)
    def test_indexer_no_match(self):
        path = '/Query[1]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) == 0)
        assert(str(xx) == path)
    def test_indexer_match(self):
        path = '/Query[0]' # returns value
        xx = p.parse(path)
        res = xx.evaluate(root)
        assert(len(res) == 1)
        assert(res[0].__class__.__name__ == "Query")
        assert(str(xx) == path)

class TestSqlDecorator:
    def test_compare_lit(self):
        path = "//ArithmeticExpression[@right > 2]"
        res = q.xpath(path)
        assert(len(res) == 2)
    def test_attrib_equal(self):
        path = "//AggFunction[@name == 'COUNT']"
        res = root.xpath(path)
        assert(len(res) == 2)
    def test_descend_attr_match(self):
        path = '//*[@left]' # returns value
        xx = p.parse(path)
        res = root.xpath(path)
        assert(len(res) >= 17)
    def test_attr_exists(self):
        path = '/Query[@soos]' # returns []
        res = root.xpath(path)
        assert(len(res) == 0)
    def test_descend_attr(self):
        path = '//@name'
        res = root.xpath(path)
        assert(len(res) >= 41)


