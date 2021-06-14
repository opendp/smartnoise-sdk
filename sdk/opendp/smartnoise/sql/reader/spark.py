from .base import Serializer, SqlReader, NameCompare
from .engine import Engine

from opendp.smartnoise._ast.tokens import Literal
from opendp.smartnoise._ast.expressions.numeric import BareFunction


class SparkReader(SqlReader):
    ENGINE = Engine.SPARK

    def __init__(self, conn):
        super().__init__(self.ENGINE)

        self.api = conn
        self.database = "Spark Session"

    def execute(self, query):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        return self.api.sql(query)

    def _to_df(rows):
        return rows

    def execute_typed(self, query):
        return self.execute(query)

    def db_name(self):
        return self.database


class SparkSerializer(Serializer):
    def serialize(self, query):
        for r_e in [n for n in query.find_nodes(BareFunction) if n.name == "RANDOM"]:
            r_e.name = "rand"

        for b in [n for n in query.find_nodes(Literal) if isinstance(n.value, bool)]:
            b.text = "'True'" if b.value else "'False'"

        return str(query)


class SparkNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]

    def identifier_match(self, query, meta):
        return self.strip_escapes(query).lower() == self.strip_escapes(meta).lower()
