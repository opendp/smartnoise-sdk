import pytest
from snsql.metadata import Metadata
from snsql.sql.parse import QueryParser

from os import listdir
from os.path import isfile, join, dirname


dir_name = dirname(__file__)
testpath = join(dir_name, "queries") + "/"

metadata = Metadata.from_file(join(dir_name, "TestDB.yaml"))

other_dirs = [f for f in listdir(testpath) if not isfile(join(testpath, f)) and f != "parse" ]

parse_files = [join(testpath + "parse/", f) for f in listdir(testpath + "parse") if isfile(join(testpath + "parse", f))]
good_files = [f for f in parse_files if not "_fail" in f]
bad_files = [f for f in parse_files if "_fail" in f]

for d in other_dirs:
    other_files = [join(testpath + d + "/", f) for f in listdir(testpath + d) if isfile(join(testpath + d, f))]
    good_files.extend(other_files)

#
#   Unit tests
#
class TestAst:
    def test_simple(self):
        query = "SELECT * FROM FOO;"
        QueryParser().parse_only(query) # try parsing without building
        qb = QueryParser().query(query)
    def test_unsupported(self):
        with pytest.raises(ValueError) as err:
            qb = QueryParser().query("SELECT * FROM FOO UNION ALL SELECT * FROM BAR", True)
    def test_bad_token(self):
        with pytest.raises(ValueError) as err:
            QueryParser().parse_only("SELECT * FROM FOO WHENCE ZIP ZAG")
        err.match("^Bad token")
    def test_batch13(self):
        qb = QueryParser().queries(open(testpath + "parse/" + "test.sql").read())
        assert len(qb) == 13
    def test_tsql_escaped_error(self):
        with pytest.raises(ValueError) as err:
            QueryParser().parse_only("SELECT [FOO.BAR] FROM HR;")
        err.match("^Lexer error")
    def test_all_good_queries(self):
        for goodpath in good_files:
            gqt = GoodQueryTester(goodpath)
            gqt.runParse()
            gqt.runBuild()
    def test_all_bad_queries(self):
        for badpath in bad_files:
            bqt = BadQueryTester(badpath)
            bqt.runParse(ValueError)
            bqt.runBuild(ValueError)

class GoodQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]
    def walk_children(self, node):
        for n in [nd for nd in node.children() if nd is not None]:
            self.walk_children(n)
    def runParse(self):
        for query in self.queries:
            try:
                QueryParser().parse_only(query)
            except Exception as e:
                raise ValueError(f"Parse error for {str(query)}: {str(e)}")
    def runBuild(self):
        for query in self.queries:
            try:
                q = QueryParser().query(query)
                self.walk_children(q)
                assert query.replace(' ','').replace('\n','').lower() == str(q).replace(' ','').replace('\n','').lower()
                self.runParseAgain(q)
            except Exception as e:
                raise ValueError(f"Parse error for {str(query)}: {str(e)}")
    def runParseAgain(self, q):
        """ Converts AST to text, re-parses to AST, and compares the two ASTs"""
        repeat = QueryParser().query(str(q))
        assert q == repeat


class BadQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]
    def runParse(self, exc):
        for query in self.queries:
            failed = False
            try:
                QueryParser().parse_only(query)
            except exc:
                failed = True
            if not failed:
                print("{0} should have thrown ValueError, but succeeded".format(query))
            assert failed
    def runBuild(self, exc):
        for query in self.queries:
            failed = False
            try:
                qb = QueryParser().query(query)
            except exc:
                failed = True
            if not failed:
                print("{0} should have thrown ValueError, but succeeded".format(query))
            assert failed
