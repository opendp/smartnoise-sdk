import pytest
from burdock.sql import QueryParser, Validate
from burdock.metadata.collection import CollectionMetadata

from os import listdir
from os.path import isfile, join, dirname

dir_name = dirname(__file__)
testpath = join(dir_name, "queries") + "/"

other_dirs = [f for f in listdir(testpath) if not isfile(join(testpath, f)) and f != "parse" and f != "validate" ]


validate_files = [join(testpath + "validate/", f) for f in listdir(testpath + "validate") if isfile(join(testpath + "validate", f))]

good_files = [f for f in validate_files if not "_fail" in f]
bad_files = [f for f in validate_files if "_fail" in f]

for d in other_dirs:
    other_files = [join(testpath + d + "/", f) for f in listdir(testpath + d) if isfile(join(testpath + d, f))]
    good_files.extend(other_files)


metadata = CollectionMetadata.from_file(join(dir_name, "Devices.yaml"))


#
#   Unit tests
#
class TestValidate:
    def test_all_good_queries(self):
        for goodpath in good_files:
            print(goodpath)
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
            print(qs)
            q = QueryParser(metadata).query(qs)
            Validate().validateQuery(q, metadata)


class BadQueryTester:
    def __init__(self, path):
        lines = open(path).readlines()
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join([line for line in lines if line.strip() != "" and not line.strip().startswith("--")])
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]

    def runValidate(self):
        for qs in self.queries:
            print(qs)
            with pytest.raises(ValueError):
                q = QueryParser(metadata).query(qs)
                self.validateSingle(q)

    def validateSingle(self, q):
        with pytest.raises(ValueError):
            Validate().validateQuery(q, metadata)
