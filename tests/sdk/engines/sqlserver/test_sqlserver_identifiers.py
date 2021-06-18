from opendp.smartnoise.sql import SqlReader

import yaml
from os import path
import keyring
import pyodbc

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
        host = conns[engine]["host"]
        port = conns[engine]["port"]
        user = conns[engine]["user"]
        conn = f"{engine}://{host}:{port}"
        password = keyring.get_password(conn, user)
        if password is not None:
            database = conns[engine]["databases"]["NameTests"]
            dsn = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host};DATABASE={database};UID={user};PWD={password}"
            print(dsn)
            connection = pyodbc.connect(dsn)
            dsn = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host};DATABASE=2NameTests;UID={user};PWD={password}"
            connection_case = pyodbc.connect(dsn)

class TestSqlServerIdentifiersNormal:
    def test_reserved(self):
        if connection is not None:
            reader = SqlReader.from_connection(connection, "postgres")
            res = reader.execute('SELECT "SELECT" FROM nametests')

class TestSqlServerIdentifiersDBCaseSensitive:
    def test_reserved(self):
        if connection_case is not None:
            reader = SqlReader.from_connection(connection_case, "postgres")
            res = reader.execute('SELECT "SELECT" FROM nametests')
