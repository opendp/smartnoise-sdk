import warnings

import pandas as pd

from .private_rewriter import Rewriter
from .private_reader import PrivateReader
from .parse import QueryParser

from .reader.base import SqlReader
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


def execute_private_query(reader, schema, budget, query):
    if not isinstance(reader, SqlReader):
        warnings.warn("[reader] API has changed to pass (reader, metadata). Please update code to pass reader first and metadata second. This will be a breaking change in future versions.", Warning)
        tmp = schema
        schema = reader
        reader = tmp
    schema = reader.metadata if hasattr(reader, "metadata") else schema
    query = reader._sanitize_query(query) if hasattr(reader, "_sanitize_query") else query
    rowset = PrivateReader(reader, schema, budget).execute(query)
    return pd.DataFrame(rowset[1:], columns=rowset[0])




