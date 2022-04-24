import os

from .base import SqlReader, NameCompare, Serializer
from .engine import Engine

from snsql._ast.ast import Relation
from snsql._ast.tokens import Literal
from snsql._ast.expression import Expression
from snsql._ast.expressions.numeric import BareFunction
from snsql._ast.expressions.sql import BooleanJoinCriteria, UsingJoinCriteria


class SqlServerReader(SqlReader):
    """
        A dumb pipe that gets a rowset back from a database using
        a SQL string, and converts types to some useful subset
    """

    ENGINE = Engine.SQL_SERVER

    def __init__(self, host=None, database=None, user=None, password=None, port=None, conn=None, **kwargs):
        super().__init__(self.ENGINE)

        if conn is not None:
            self.conn = conn
        else:
            # generate a connection string
            self.host = host
            self.database = database
            self.user = user
            self.port = port
            try:
                import pyodbc

                # this reader would break in the module code as is
                self.api = pyodbc

                if password is None:
                    if "SA_PASSWORD" in os.environ:
                        password = os.environ["SA_PASSWORD"]
                self.password = password

                self.update_connection_string()
            except:
                pass

    def execute(self, query, *ignore, accuracy:bool=False):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        if self.conn is not None:
            cnxn = self.conn
        else:
            cnxn = self.api.connect(self.connection_string)
        cursor = cnxn.cursor()
        cursor.execute(str(query))
        if cursor.description is None:
            return []
        else:
            col_names = [tuple(desc[0] for desc in cursor.description)]
            rows = [row for row in cursor]
            return col_names + rows

    def update_connection_string(self):
        self.connection_string = "Server={0}{1};UID={2}".format(
            self.host, "" if self.port is None else "," + str(self.port), self.user
        )
        self.connection_string += (
            ";Database={0}".format(self.database) if self.database is not None else ""
        )
        self.connection_string += (
            ";PWD={0}".format(self.password) if self.password is not None else ""
        )
        self.connection_string = "Driver={ODBC Driver 17 for SQL Server};" + self.connection_string

    def switch_database(self, dbname):
        sql = "USE " + dbname + ";"
        self.execute(sql)

    def db_name(self):
        sql = "SELECT DB_NAME();"
        dbname = self.execute(sql)[1][0]
        return dbname


class SqlServerSerializer(Serializer):
    def serialize(self, query):
        for re in [n for n in query.find_nodes(BareFunction) if n.name == "RANDOM"]:
            re.name = "NEWID"

        for b in [n for n in query.find_nodes(Literal) if isinstance(n.value, bool)]:
            b.text = "'TRUE'" if b.value else "'FALSE'"

        # T-SQL doesn't support USING critera, rewrite to ON
        for rel in [
            n for n in query.find_nodes(Relation) if n.joins is not None and len(n.joins) > 0
        ]:
            join_idx = 0
            for j in [j for j in rel.joins if isinstance(j.criteria, UsingJoinCriteria)]:
                join_idx += 1
                ids = j.criteria.identifiers
                if rel.primary.alias is None:
                    rel.primary.alias = "PJXX"  # should use a name scope manager here
                if j.right.alias is None:
                    j.right.alias = "PJYY" + str(join_idx)  # should use naming scope

                left = [rel.primary.alias + "." + str(i) for i in ids]
                right = [j.right.alias + "." + str(i) for i in ids]
                frag = " AND ".join(l + " = " + r for l, r in zip(left, right))
                j.criteria = BooleanJoinCriteria(Expression(frag))

        return str(query)


class SqlServerNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]

    def identifier_match(self, from_query, from_meta):
        return self.strip_escapes(from_query).lower() == self.strip_escapes(from_meta).lower()
