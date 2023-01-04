import os

from .base import SqlReader, NameCompare, Serializer
from .engine import Engine


class MySqlReader(SqlReader):
    """
        A dumb pipe that gets a rowset back from a database using
        a SQL string, and converts types to some useful subset
    """

    ENGINE = Engine.MYSQL

    def __init__(self, conn, **kwargs):
        super().__init__(self.ENGINE)
        if conn is None:
            raise ValueError("Please pass a pymysql connection to the MySqlReader")
        self.conn = conn

    def execute(self, query, *ignore, accuracy:bool=False):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        cnxn = self.conn
        try:
            cursor = cnxn.cursor()
            cursor.execute(str(query))
            cnxn.commit()
            if cursor.description is None:
                return []
            else:
                col_names = [tuple(desc[0] for desc in cursor.description)]
                rows = [row for row in cursor]
                return col_names + rows
        except Exception as e:
            cnxn.rollback()
            raise e

class MySqlNameCompare(NameCompare):
    def __init__(self, search_path=None):
        # mysql has no default schema
        self.search_path = search_path if search_path is not None else []

    def identifier_match(self, from_query, from_meta):
        from_query = self.clean_escape(from_query)
        from_meta = self.clean_escape(from_meta)
        if from_query == from_meta:
            return True
        if self.is_escaped(from_meta) and from_meta.lower() == from_meta:
            from_meta = from_meta.lower().replace('"', "")
        if self.is_escaped(from_query) and from_query.lower() == from_query:
            from_query = from_query.lower().replace('"', "")
        return from_meta == from_query

    def should_escape(self, identifier):
        if self.is_escaped(identifier):
            return False
        if identifier.lower() in self.reserved():
            return True
        if identifier.replace(" ", "") == identifier.lower():
            return False
        else:
            return True

class MySqlSerializer(Serializer):
    def __init__(self):
        super().__init__()
    def serialize(self, ast):
        query = str(ast)
        query = query.replace("RANDOM ( )", "RAND()")
        return query
