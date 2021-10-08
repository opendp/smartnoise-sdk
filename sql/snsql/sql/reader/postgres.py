import os

from .base import SqlReader, NameCompare, Serializer
from .engine import Engine


class PostgresReader(SqlReader):
    """
        A dumb pipe that gets a rowset back from a database using
        a SQL string, and converts types to some useful subset
    """

    ENGINE = Engine.POSTGRES

    def __init__(self, host=None, database=None, user=None, password=None, port=None, conn=None, **kwargs):
        super().__init__(self.ENGINE)

        self.conn = None
        if conn is not None:
            self.conn = conn
        else:
            # generate a connection string
            self.host = host
            self.database = database
            self.user = user
            self.port = port

            if password is None:
                if "POSTGRES_PASSWORD" in os.environ:
                    password = os.environ["POSTGRES_PASSWORD"]
            self.password = password
            self._update_connection_string()
            try:
                import psycopg2
                self.api = psycopg2
            except:
                pass

    def execute(self, query, *ignore, accuracy:bool=False):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        cnxn = self.conn
        if cnxn is None:
            cnxn = self.api.connect(self.connection_string)
        cursor = cnxn.cursor()
        cursor.execute(str(query))
        if cursor.description is None:
            return []
        else:
            col_names = [tuple(desc[0] for desc in cursor.description)]
            rows = [row for row in cursor]
            return col_names + rows

    def _update_connection_string(self):
        self.connection_string = "user='{0}' host='{1}'".format(self.user, self.host)
        self.connection_string += (
            " dbname='{0}'".format(self.database) if self.database is not None else ""
        )
        self.connection_string += " port={0}".format(self.port) if self.port is not None else ""
        self.connection_string += (
            " password='{0}'".format(self.password) if self.password is not None else ""
        )

    def switch_database(self, dbname):
        sql = "\\c " + dbname
        self.execute(sql)

class PostgresNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["public"]

    def identifier_match(self, query, meta):
        query = self.clean_escape(query)
        meta = self.clean_escape(meta)
        if query == meta:
            return True
        if self.is_escaped(meta) and meta.lower() == meta:
            meta = meta.lower().replace('"', "")
        if self.is_escaped(query) and query.lower() == query:
            query = query.lower().replace('"', "")
        return meta == query

    def should_escape(self, identifier):
        if self.is_escaped(identifier):
            return False
        if identifier.lower() in self.reserved():
            return True
        if identifier.replace(" ", "") == identifier.lower():
            return False
        else:
            return True

class PostgresSerializer(Serializer):
    def __init__(self):
        super().__init__()

