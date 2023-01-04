from snsql.sql.reader.base import SqlReader

connection = None # database "NameTests"
connection_case = None # database "2NameTests"

class TestSqlServerIdentifiersNormal:
    def test_reserved(self, test_databases):
        connection = test_databases.get_connection(database="NameTests", engine="sqlserver")
        if connection is not None:
            reader = SqlReader.from_connection(connection, "sqlserver")
            res = reader.execute('SELECT "SELECT" FROM nametests')

class TestSqlServerIdentifiersDBCaseSensitive:
    def test_reserved(self, test_databases):
        connection_case = test_databases.get_connection(database="2NameTests", engine="sqlserver")
        if connection_case is not None:
            reader = SqlReader.from_connection(connection_case, "sqlserver")
            res = reader.execute('SELECT "SELECT" FROM nametests')
