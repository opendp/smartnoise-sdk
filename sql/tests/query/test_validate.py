import pytest
from snsql._ast.validate import Validate
from snsql.sql.parse import QueryParser
from snsql.metadata import Metadata

from os import listdir
from os.path import isfile, join, dirname

dir_name = dirname(__file__)
testpath = join(dir_name, "queries") + "/"

other_dirs = [f for f in listdir(testpath) if not isfile(join(testpath, f)) and f not in ["parse", "validate", "validate_pums", "compare"]]


validate_files = [join(testpath + "validate/", f) for f in listdir(testpath + "validate") if isfile(join(testpath + "validate", f))]

good_files = [f for f in validate_files if not "_fail" in f]
bad_files = [f for f in validate_files if "_fail" in f]

for d in other_dirs:
    other_files = [join(testpath + d + "/", f) for f in listdir(testpath + d) if isfile(join(testpath + d, f))]
    good_files.extend(other_files)


metadata = Metadata.from_file(join(dir_name, "Devices.yaml"))


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
            try:
                q = QueryParser(metadata).query(qs)
                Validate().validateQuery(q, metadata)
            except Exception as e:
                raise ValueError(f"Parse and validate failed for query: {str(q)}")


class BadQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]

    def runValidate(self):
        for qs in self.queries:
            with pytest.raises(ValueError):
                q = QueryParser(metadata).query(qs)
                self.validateSingle(q)

    def validateSingle(self, q):
        with pytest.raises(ValueError):
            Validate().validateQuery(q, metadata)
