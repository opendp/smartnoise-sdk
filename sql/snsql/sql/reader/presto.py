import os
from sqlite3.dbapi2 import connect

from snsql._ast.ast import Relation
from snsql._ast.tokens import Literal
from snsql._ast.expression import Expression
from snsql._ast.expressions.numeric import BareFunction
from snsql._ast.expressions.sql import BooleanJoinCriteria, UsingJoinCriteria
from .base import SqlReader, NameCompare, Serializer
from .engine import Engine


class PrestoReader(SqlReader):
    """
        A dumb pipe that gets a rowset back from a database using
        a SQL string, and converts types to some useful subset
    """

    ENGINE = Engine.PRESTO

    def __init__(self, host=None, database=None, user=None, password=None, port=None, conn=None, **kwargs):
        super().__init__(self.ENGINE)

        if conn is not None:
            self.conn = conn
    
        self.host = host
        self.database = database
        self.user = user
        if port is not None:
            self.port = int(port)

        if password is None:
            if "PRESTO_PASSWORD" in os.environ:
                password = os.environ["PRESTO_PASSWORD"]
        self.password = password

        self.update_connection_string()

    def execute(self, query, *ignore, accuracy:bool=False):
        """
            Executes a raw SQL string against the database and returns
            tuples for rows.  This will NOT fix the query to target the
            specific SQL dialect.  Call execute_typed to fix dialect.
        """
        import prestodb
        self.api = prestodb.dbapi

        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        if self.conn is not None:
            cnxn = self.conn
        else:
            cnxn = self.api.connect(
                host=self.host,
                http_scheme="https" if self.port == 443 else "http",
                user=self.user,
                port=self.port,
                catalog=self.database,
            )
        cursor = cnxn.cursor()
        cursor.execute(str(query).replace(";", ""))
        rows = cursor.fetchall()
        if cursor.description is None:
            return []
        else:
            col_names = [tuple(desc[0] for desc in cursor.description)]
            rows = [row for row in rows]
            return col_names + rows

    def update_connection_string(self):
        self.connection_string = None
        pass

    def switch_database(self, dbname):
        sql = "USE " + dbname + ";"
        self.execute(sql)

    def db_name(self):
        return self.database


class PrestoNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]

    def identifier_match(self, query, meta):
        return self.strip_escapes(query).lower() == self.strip_escapes(meta).lower()

class PrestoSerializer(Serializer):
    def __init__(self):
        super().__init__()