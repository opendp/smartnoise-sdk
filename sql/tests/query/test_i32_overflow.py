import os
import subprocess
import copy
from snsql import Privacy, from_connection
import pandas as pd

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

class TestIntegerOverflow:
    def test_i32_overflow(self):
        privacy = Privacy(epsilon=1.0, delta=1e-5)
        pums = pd.read_csv(csv_path)
        conn = from_connection(pums, metadata=meta_path, privacy=privacy)

        query = "SELECT AVG(income) FROM PUMS.PUMS"

        query_ast = conn.parse_query_string(query)
        from_clause = copy.deepcopy(query_ast.source)

        subquery_ast, _ = conn._rewrite(query)

        subquery_ast.source = from_clause
        subquery = str(subquery_ast)

        inner = conn.execute(subquery)

        # scale the result to cause overflow for i32
        scale = 2**36 / (500_000 * 1000)
        inner[1][0] = inner[1][0] * scale
        inner[1][1] = inner[1][1] * scale
        assert(inner[1][1] > 2**32)
        _ = conn._execute_ast(query_ast, pre_aggregated=inner)
