import importlib

#from snsql.metadata import Metadata
from .base import SqlReader, NameCompare, Serializer
from .engine import Engine
import copy
import warnings
import re


class SQLiteReader(SqlReader):
    ENGINE = Engine.SQLITE

    def __init__(self, conn, **kwargs):
        if conn is None:
            raise ValueError("Pass in a SQLite connection")
        self.conn = conn
        super().__init__(self.ENGINE)
        import sqlite3
        ver = [int(part) for part in sqlite3.sqlite_version.split(".")]
        if len(ver) == 3:
            # all historical versions of SQLite have 3 parts
            if (
                ver[0] < 3
                or (ver[0] == 3 and ver[1] < 2)
                or (ver[0] == 3 and ver[1] == 2 and ver[2] < 6)
            ):
                warnings.warn(
                    "This python environment has outdated sqlite version {0}.  SQLiteReader will fail on queries that use private_key.  Please upgrade to a newer Python environment (with sqlite >= 3.2.6), or ensure that you only use row_privacy.".format(
                        sqlite3.sqlite_version
                    ),
                    Warning,
                )


    def execute(self, query, *ignore, accuracy:bool=False):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        try:
            cursor = self.conn.cursor()
            cursor.execute(str(query))
            self.conn.commit()
            if cursor.description is None:
                return []
            else:
                col_names = [tuple(desc[0] for desc in cursor.description)]
                rows = [row for row in cursor]
                return col_names + rows
        except Exception as e:
            self.conn.rollback()
            raise e

class SQLiteNameCompare(NameCompare):
    def __init__(self, search_path=None):
        super().__init__(search_path)

class SQLiteSerializer(Serializer):
    def __init__(self):
        super().__init__()
