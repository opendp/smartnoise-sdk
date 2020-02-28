from burdock.metadata.collection import CollectionMetadata
from .private_rewrite import Rewriter
from .private_reader import PrivateReader
from burdock.reader.sql.pandas import PandasReader
from .parse import QueryParser


def execute_private_query(reader, schema, budget, query):
    schema = reader.metadata if hasattr(reader, "metadata") else schema
    query = reader._sanitize_query(query) if hasattr(reader ,"_sanitize_query") else query
    return PrivateReader(reader, schema, budget).execute(query)

__all__ = ["CollectionMetadata", "Rewriter", "QueryParser",
           "PandasReader", "execute_private_query"]
