import os

from burdock.metadata.name_compare import BaseNameCompare
from .rowset import TypedRowset
from burdock.query.sql.ast.ast import Relation
from burdock.query.sql.ast.tokens import Literal
from burdock.query.sql.ast.expression import Expression
from burdock.query.sql.ast.expressions.numeric import BareFunction
from burdock.query.sql.ast.expressions.sql import BooleanJoinCriteria, UsingJoinCriteria

"""
    A dumb pipe that gets a rowset back from a database using 
    a SQL string, and converts types to some useful subset
"""
class DataReader:
    def __init__(self, host, database, user, password=None, port=None):
        import prestodb
        self.api = prestodb.dbapi
        self.engine = "Presto"

        self.host = host
        self.database = database
        self.user = user
        self.port = int(port)

        if password is None:
            if 'PRESTO_PASSWORD' in os.environ:
                password = os.environ['PRESTO_PASSWORD']
        self.password = password

        self.update_connection_string()
        self.serializer = None
        self.compare = PrestoNameCompare()
    """
        Executes a raw SQL string against the database and returns
        tuples for rows.  This will NOT fix the query to target the
        specific SQL dialect.  Call execute_typed to fix dialect.
    """
    def execute(self, query):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        cnxn = self.api.connect(
            host=self.host,
            http_scheme='https' if self.port == 443 else 'http',
            user=self.user,
            port=self.port,
            catalog=self.database
        )
        cursor = cnxn.cursor()
        cursor.execute(str(query).replace(';',''))
        rows = cursor.fetchall()
        if cursor.description is None:
            return []
        else:
            col_names = [tuple(desc[0] for desc in cursor.description)]
            rows = [row for row in rows]
            return col_names + rows

    """
        Executes a parsed AST and returns a typed recordset.
        Will fix to target approprate dialect. Needs symbols.
    """
    def execute_typed(self, query):
        if isinstance(query, str):
            raise ValueError("Please pass ASTs to execute_typed.  To execute strings, use execute.")

        syms = query.all_symbols()
        types = [s[1].type() for s in syms]
        sens = [s[1].sensitivity() for s in syms]

        if hasattr(self, 'serializer') and self.serializer is not None:
            query_string = self.serializer.serialize(query)
        else:
            query_string = str(query)
        rows = self.execute(query_string)
        return TypedRowset(rows, types, sens)

    def update_connection_string(self):
        self.connection_string = None
        pass

    def switch_database(self, dbname):
        sql = "USE " + dbname + ";"
        self.execute(sql)

    def db_name(self):
        return self.database

class PrestoNameCompare(BaseNameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]
    def identifier_match(self, query, meta):
        return self.strip_escapes(query).lower() == self.strip_escapes(meta).lower()