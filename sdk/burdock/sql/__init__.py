from burdock.metadata.collection import CollectionMetadata
from .ast.parse import QueryParser
from .ast.validate import Validate
from .private_rewrite import Rewriter
from .private_query import PrivateReader
from burdock.reader.sql.pandas import PandasReader


def execute_private_query(reader, schema, budget, query):
    schema = reader.metadata if hasattr(reader, "metadata") else schema
    query = reader._sanitize_query(query) if hasattr(reader ,"_sanitize_query") else query
    return PrivateReader(reader, schema, budget).execute(query)

__all__ = ["CollectionMetadata", "QueryParser", "Rewriter", "Validate",
           "PandasReader", "execute_private_query"]
