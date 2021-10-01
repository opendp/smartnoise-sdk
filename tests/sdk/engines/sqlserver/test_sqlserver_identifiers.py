from opendp.smartnoise.sql import SqlReader

import yaml
from os import path

connection = None # database "NameTests"
connection_case = None # database "2NameTests"

home = path.expanduser("~")
p = path.join(home, ".smartnoise", "connections-unit.yaml")
if not path.exists(p):
    print ("No config file at ~/.smartnoise/connections-unit.yaml")
else:
    with open(p, 'r') as stream:
        conns = yaml.safe_load(stream)
    if conns is None:
        pass
    else:
        engine = "sqlserver"
        has_sqlserver = False
        try:
            host = conns[engine]["host"]
            port = conns[engine]["port"]
            user = conns[engine]["user"]
            conn = f"{engine}://{host}:{port}"
            import keyring
            import pyodbc
            password = keyring.get_password(conn, user)
            if password is not None:
                database = conns[engine]["databases"]["NameTests"]
                dsn = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host};DATABASE={database};UID={user};PWD={password}"
                connection = pyodbc.connect(dsn)
                dsn = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host};DATABASE=2NameTests;UID={user};PWD={password}"
                connection_case = pyodbc.connect(dsn)
                has_sqlserver = True
        except:
            has_sqlserver = False

class TestSqlServerIdentifiersNormal:
    def test_reserved(self):
        if connection is not None and has_sqlserver:
            reader = SqlReader.from_connection(connection, "postgres")
            res = reader.execute('SELECT "SELECT" FROM nametests')

class TestSqlServerIdentifiersDBCaseSensitive:
    def test_reserved(self):
        if connection_case is not None and has_sqlserver:
            reader = SqlReader.from_connection(connection_case, "postgres")
            res = reader.execute('SELECT "SELECT" FROM nametests')
