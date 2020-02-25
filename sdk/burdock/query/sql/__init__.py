from burdock.metadata.config_file import MetadataLoader
from .ast.parse import QueryParser
from .ast.validate import Validate
from .private.rewrite import Rewriter
from .private.query import PrivateQuery
from .reader.pandas_reader import PandasReader


def execute_private_query(reader, schema, budget, query):
    schema = reader.metadata if hasattr(reader, "metadata") else schema
    query = reader._sanitize_query(query) if hasattr(reader ,"_sanitize_query") else query
    return PrivateQuery(reader, schema, budget).execute(query)

__all__ = ["MetadataLoader", "QueryParser", "Rewriter", "Validate",
           "PandasReader", "execute_private_query"]
