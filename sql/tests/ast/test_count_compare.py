import os
import subprocess
from snsql.metadata import Metadata
from snsql.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
meta = Metadata.from_file(meta_path)


class TestCountCompare:
    def test_count_star_first(self):
        query = 'SELECT COUNT(*) AS c, COUNT(sex) AS s, COUNT(age) AS a FROM PUMS.PUMS'
        ast = QueryParser().query(query)
        count_star = ast.select.namedExpressions[0].expression
        count_sex = ast.select.namedExpressions[1].expression
        count_age = ast.select.namedExpressions[2].expression

        assert count_star != count_sex
        assert count_sex != count_star
        assert count_sex != count_age

    def test_count_star_last(self):
        query = 'SELECT COUNT(sex) AS s, COUNT(age) AS a, COUNT(*) AS c FROM PUMS.PUMS'
        ast = QueryParser().query(query)
        count_sex = ast.select.namedExpressions[0].expression
        count_age = ast.select.namedExpressions[1].expression
        count_star = ast.select.namedExpressions[2].expression

        assert count_star != count_sex
        assert count_sex != count_star
        assert count_sex != count_age
    def test_count_star_first_symbols(self):
        query = 'SELECT COUNT(*) AS c, COUNT(sex) AS s, COUNT(age) AS a FROM PUMS.PUMS'
        ast = QueryParser(meta).query(query)
        count_star = ast._select_symbols[0].expression
        count_sex = ast._select_symbols[1].expression
        count_age = ast._select_symbols[2].expression

        assert count_star != count_sex
        assert count_sex != count_star
        assert count_sex != count_age
    def test_count_star_last_symbols(self):
        query = 'SELECT COUNT(sex) AS s, COUNT(age) AS a, COUNT(*) AS c FROM PUMS.PUMS'
        ast = QueryParser(meta).query(query)
        count_sex = ast._select_symbols[0].expression
        count_age = ast._select_symbols[1].expression
        count_star = ast._select_symbols[2].expression

        assert count_star != count_sex
        assert count_sex != count_star
        assert count_sex != count_age


