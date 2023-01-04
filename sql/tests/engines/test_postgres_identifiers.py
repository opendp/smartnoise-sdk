from snsql.sql.reader.base import SqlReader

connection = None # database "NameTests"
connection_case = None # database "2NameTests"

class TestPostgresIdentifiersNormal:
    def test_reserved(self, test_databases):
        connection = test_databases.get_connection(database="NameTests", engine="postgres")
        if connection is not None:
            reader = SqlReader.from_connection(connection, "postgres")
            res = reader.execute('SELECT "select" FROM nametests')

class TestPostgresIdentifiersDBCaseSensitive:
    def test_reserved(self, test_databases):
        connection_case = test_databases.get_connection(database="2NameTests", engine="postgres")
        if connection_case is not None:
            reader = SqlReader.from_connection(connection_case, "postgres")
            res = reader.execute('SELECT "select" FROM nametests')
