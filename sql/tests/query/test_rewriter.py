from os import listdir
from os.path import isfile, join, dirname

from snsql.metadata import Metadata
from snsql.sql.private_rewriter import Rewriter
from snsql.sql.parse import QueryParser


dir_name = dirname(__file__)
testpath = join(dir_name, "queries") + "/"

other_dirs = [f for f in listdir(testpath) if not isfile(join(testpath, f)) and f not in ["parse", "compare", "validate", "rewrite", "validate_pums"] ]
rewrite_files = [join(testpath + "rewrite/", f) for f in listdir(testpath + "rewrite") if isfile(join(testpath + "rewrite", f))]

good_files = [f for f in rewrite_files if not "_fail" in f]
bad_files = [f for f in rewrite_files if "_fail" in f]

for d in other_dirs:
    other_files = [join(testpath + d + "/", f) for f in listdir(testpath + d) if isfile(join(testpath + d, f))]
    good_files.extend(other_files)

metadata = Metadata.from_file(join(dir_name, "Devices.yaml"))


#
#   Unit tests
#
class TestRewrite:
    def test_all_good_queries(self):
        for goodpath in good_files:
            gqt = GoodQueryTester(goodpath)
            gqt.runRewrite()

    def test_all_bad_queries(self):
        for badpath in bad_files:
            bqt = BadQueryTester(badpath)
            bqt.runRewrite()

class GoodQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]

    def runRewrite(self):
        qb = QueryParser(metadata).queries(self.queryBatch)
        for q in qb:
            try:
                new_q = Rewriter(metadata).query(q)
                assert q.has_symbols()
                assert new_q.has_symbols()
                assert all([qt.expression.type() == nqt.expression.type() for qt, nqt in zip(q._select_symbols, new_q._select_symbols) ])
            except Exception as e:
                raise ValueError(f"Rewrite error for query: {str(q)}")

class BadQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]

    def runRewrite(self):
        pass
