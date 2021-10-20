from snsql.sql.reader.postgres import PostgresNameCompare
import pytest
from snsql._ast.validate import Validate
from snsql.sql.parse import QueryParser
from snsql.metadata import Metadata

from os import listdir
from os.path import isfile, join, dirname
import subprocess

"""Unit test driver for testing valid and invalid queries that
    use the PUMS schema.

All queries in tests/validate_pums.sql should pass validation
All queries in tests/validate_pums_fail.sql should parse and build AST, but fail validation

In other words, these tests should catch simple cases of valid SQL
    that violates differential privacy rules, which the validator is
    expected to prevent.

Add new queries to these two SQL files to test edge cases.

Note that the other validate test suite does a cumulative test pass using more
    complex schema, with each increasing level (parse->ast->validate->rewrite)
    running all queries for the levels before.  Because those test suites don't use
    PUMS schema, we do not do the cumulative test here, but it is done in the full
    unit test pass.
"""

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = join(git_root_dir, join("datasets", "PUMS_pid.yaml"))

dir_name = dirname(__file__)
testpath = join(dir_name, "queries") + "/"

validate_files = [join(testpath + "validate_pums/", f) for f in listdir(testpath + "validate_pums") if isfile(join(testpath + "validate_pums", f))]

good_files = [f for f in validate_files if not "_fail" in f]
bad_files = [f for f in validate_files if "_fail" in f]

metadata = Metadata.from_file(meta_path)
metadata.compare = PostgresNameCompare()


#
#   Unit tests
#
class TestValidate:
    def test_all_good_queries(self):
        for goodpath in good_files:
            gqt = GoodQueryTester(goodpath)
            gqt.runValidate()
    def test_all_bad_queries(self):
        for badpath in bad_files:
            bqt = BadQueryTester(badpath)
            bqt.runValidate()


class GoodQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]

    def runValidate(self):
        for qs in self.queries:
            q = QueryParser(metadata).query(qs)
            try:
                Validate().validateQuery(q, metadata)
            except:
                raise ValueError(f"Validation failed for query: {str(q)}")


class BadQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]

    def runValidate(self):
        for qs in self.queries:
            q = QueryParser(metadata).query(qs)
            self.validateSingle(q)

    def validateSingle(self, q):
        try:
            Validate().validateQuery(q, metadata)
        except ValueError:
            return
        raise ValueError(f"Query didn't fail validation: {str(q)}")
