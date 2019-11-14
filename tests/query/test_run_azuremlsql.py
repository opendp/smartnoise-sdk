from burdock.query.sql.private.query import PrivateQuery
from burdock.query.sql import QueryParser, MetadataLoader, Rewriter
from azureml_reader import DataReader

from azureml.core import Workspace


from os import listdir
from os.path import isfile, join, dirname

test_file_dir = dirname(__file__)

db = MetadataLoader(join(test_file_dir, "Simple.yaml")).read_schema()


reader = DataReader(Workspace.from_config().datastores["burdock_demo"])
private_reader = PrivateQuery(reader, db)


#
#   Unit tests
#
class TestAzureMLSql:
    def test_all_good_queries(self):
        gqt = GoodQueryTester()
        gqt.runRewrite()


class GoodQueryTester:
    def __init__(self):
        lines = ["SELECT COUNT(my_column) as my_count from dbo.test_table;",
                 "SELECT AVG(my_column) as my_avg from dbo.test_table;"]
        self.queryBatch = "\n".join(lines)
        queryLines = " ".join(lines)
        self.queries = [q.strip() for q in queryLines.split(";") if q.strip() != ""]

    def runRewrite(self):
        qb = QueryParser(db).queries(self.queryBatch)
        for q in qb:
            new_q = Rewriter(db).query(q)
            orig = reader.execute_typed(q)
            assert len(orig) > 0
            rewritten = reader.execute_typed(new_q)
            assert len(rewritten) > 0
            assert len(rewritten[0]) == len(orig[0])  # must have same number of columns
            private = private_reader.execute(str(q))
            assert len(private) > 0
            assert len(private[0]) == len(orig.colnames)  # must have same number of columns
