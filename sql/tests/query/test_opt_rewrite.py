import os
from snsql._ast.tokens import unique
import subprocess
from snsql._ast.tokens import Column
import pandas as pd

from snsql import *
from snsql.metadata import Metadata, Table, Int
from snsql.sql import PrivateReader
from snsql.sql.reader.pandas import PandasReader
from snsql.sql.reader.postgres import PostgresReader
from snsql.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))



class TestOptRewriter:
    def test_with_censor_dims(self):
        meta = Metadata.from_file(meta_path)
        df = pd.read_csv(csv_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, privacy=Privacy(epsilon=3.0))
        query = "SELECT COUNT (*) AS foo, COUNT(DISTINCT pid) AS bar FROM PUMS.PUMS"
        q = QueryParser(meta).query(query)
        inner, outer = private_reader._rewrite_ast(q)
        ne = outer.select.namedExpressions
        assert(ne[0].expression.expression.name != 'keycount')
        assert(ne[1].expression.expression.name == 'keycount')
    def test_with_row_privacy(self):
        meta = Metadata.from_file(meta_path)
        meta['PUMS.PUMS'].row_privacy = True
        meta['PUMS.PUMS']['pid'].is_key = False
        df = pd.read_csv(csv_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, privacy=Privacy(epsilon=3.0))
        query = "SELECT COUNT (*) AS foo, COUNT(DISTINCT pid) AS bar FROM PUMS.PUMS"
        q = QueryParser(meta).query(query)
        inner, outer = private_reader._rewrite_ast(q)
        ne = outer.select.namedExpressions
        assert(ne[0].expression.expression.name == 'keycount')
        assert(ne[1].expression.expression.name != 'keycount')
    def test_case_sensitive(self):
        sample = Table("PUMS", "PUMS", [
            Int('pid', is_key=True),
            Int('"PiD"')
        ], 150)
        meta = Metadata([sample], "csv")
        reader = PostgresReader("localhost", "PUMS", "admin", "password")
        private_reader = PrivateReader(reader, meta, privacy=Privacy(epsilon=3.0))
        query = 'SELECT COUNT (DISTINCT pid) AS foo, COUNT(DISTINCT "PiD") AS bar FROM PUMS.PUMS'
        inner, outer = private_reader._rewrite(query)
        ne = outer.select.namedExpressions
        assert(ne[0].expression.expression.name == 'keycount')
        assert(ne[1].expression.expression.name != 'keycount')
    def test_reuse_expression(self):
        meta = Metadata.from_file(meta_path)
        df = pd.read_csv(csv_path)
        reader = PandasReader(df, meta)
        private_reader = PrivateReader(reader, meta, privacy=Privacy(epsilon=3.0))
        query = 'SELECT AVG(age), SUM(age), COUNT(age) FROM PUMS.PUMS'
        q = QueryParser(meta).query(query)
        inner, outer = private_reader._rewrite(query)
        names = unique([f.name for f in outer.select.namedExpressions.find_nodes(Column)])
        assert(len(names) == 2)
        assert('count_age' in names)
        assert('sum_age' in names)
