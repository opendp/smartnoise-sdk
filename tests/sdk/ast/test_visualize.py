import os
import subprocess
import copy
import pytest

import pandas as pd
from pandasql import sqldf
import math

from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.sql import PrivateReader, PandasReader
from opendp.smartnoise.sql.parse import QueryParser
from graphviz import Digraph
from opendp.smartnoise._ast.ast import Query, Table
from opendp.smartnoise._ast.expressions.sql import AggFunction

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

#   Unit tests
#
class TestAstVisualize:
    def test_viz_query(self):
        query = "SELECT SUM(age) AS my_sum FROM pums.pums GROUP BY age"
        parsed_query = QueryParser().query(query)
        graph = parsed_query.visualize(color_types={Query:'red'}, n_trunc=30)
        assert(isinstance(graph, Digraph))
        #graph.render('ast_digraph', view=True, cleanup=True)
    def test_viz_query_symbols(self):
        query = "SELECT SUM(age) AS my_sum FROM PUMS.PUMS GROUP BY age"
        parsed_query = QueryParser(schema).query(query)
        graph = parsed_query.visualize(color_types={Table:'red'}, n_trunc=5)
        assert(isinstance(graph, Digraph))
        #graph.render('ast_digraph', view=True, cleanup=True)
    def test_viz_query_rewritten(self):
        query = "SELECT SUM(age) AS my_sum FROM PUMS.PUMS GROUP BY age"
        parsed_query = QueryParser(schema).query(query)
        reader = PandasReader(df, schema)
        private_reader = PrivateReader(reader, schema, 1.0)
        inner, outer = private_reader.rewrite_ast(parsed_query)
        graph = outer.visualize(n_trunc=30)
        assert(isinstance(graph, Digraph))
        #graph.render('ast_digraph', view=True, cleanup=True)
        graph = inner.visualize(n_trunc=30)
        assert(isinstance(graph, Digraph))
        #graph.render('ast_digraph', view=True, cleanup=True)
    def test_viz_child_nodes(self):
        query = "SELECT AVG(age) AS my_sum FROM PUMS.PUMS GROUP BY age"
        reader = PandasReader(df, schema)
        private_reader = PrivateReader(reader, schema, 1.0)
        inner, outer = private_reader.rewrite(query)
        aggfuncs = outer.find_nodes(AggFunction)
        for aggfunc in aggfuncs:
            graph = aggfunc.visualize(n_trunc=30)
            assert(isinstance(graph, Digraph))
            #graph.render('ast_digraph', view=True, cleanup=True)
