
from .private_rewriter import Rewriter
from .private_reader import PrivateReader
from .parse import QueryParser

from .reader.pandas import PandasReader
from .reader.presto import PrestoReader
from .reader.postgres import PostgresReader
from .reader.sql_server import SqlServerReader
from .reader.spark import SparkReader

__all__ = ["PandasReader",
           "PostgresReader",
           "PrestoReader",
           "SqlServerReader",
           "SparkReader",
           "Rewriter",
           "QueryParser",
           "execute_private_query"]


def execute_private_query(schema, reader, budget, query):
    schema = reader.metadata if hasattr(reader, "metadata") else schema
    query = reader._sanitize_query(query) if hasattr(reader, "_sanitize_query") else query
    return PrivateReader(schema, reader, budget).execute(query)


