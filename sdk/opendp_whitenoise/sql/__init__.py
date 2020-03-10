
from .private_rewriter import Rewriter
from .private_reader import PrivateReader
from .parse import QueryParser


def execute_private_query(reader, schema, budget, query):
    schema = reader.metadata if hasattr(reader, "metadata") else schema
    query = reader._sanitize_query(query) if hasattr(reader ,"_sanitize_query") else query
    return PrivateReader(reader, schema, budget).execute(query)


__all__ = ["Rewriter", "QueryParser", "execute_private_query"]
