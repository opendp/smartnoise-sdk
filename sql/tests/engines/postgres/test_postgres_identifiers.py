from snsql.sql import SqlReader

import yaml
from os import path

has_postgres = False
try:
    import keyring
    import psycopg2
    has_postgres = True
except:
    pass

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
        engine = "postgres"
        host = conns[engine]["host"]
        port = conns[engine]["port"]
        user = conns[engine]["user"]
        conn = f"{engine}://{host}:{port}"
        if has_postgres:
            password = keyring.get_password(conn, user)
            if password is not None:
                database = conns[engine]["databases"]["NameTests"]
                connection = psycopg2.connect(database=database, host=host, user=user, port=port, password=password)
                connection_case = psycopg2.connect(database="2NameTests", host=host, user=user, port=port, password=password)


class TestPostgresIdentifiersNormal:
    def test_reserved(self):
        if connection is not None and has_postgres:
            reader = SqlReader.from_connection(connection, "postgres")
            res = reader.execute('SELECT "select" FROM nametests')

class TestPostgresIdentifiersDBCaseSensitive:
    def test_reserved(self):
        if connection_case is not None and has_postgres:
            reader = SqlReader.from_connection(connection_case, "postgres")
            res = reader.execute('SELECT "select" FROM nametests')
