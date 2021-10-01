from .base import Serializer, SqlReader, NameCompare
from .engine import Engine

from snsql._ast.tokens import FuncName, Literal
from snsql._ast.expressions.numeric import BareFunction


class SparkReader(SqlReader):
    ENGINE = Engine.SPARK

    def __init__(self, conn, **kwargs):
        super().__init__(self.ENGINE)

        self.api = conn
        self.database = "Spark Session"

    def execute(self, query, *ignore, accuracy:bool=False):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        res = self.api.sql(query)
        return res

    def _to_df(rows):
        return rows

    def db_name(self):
        return self.database


class SparkSerializer(Serializer):
    def serialize(self, query):
        for r_e in [n for n in query.find_nodes(BareFunction) if n.name == "RANDOM"]:
            r_e.name = FuncName("rand")

        for b in [n for n in query.find_nodes(Literal) if isinstance(n.value, bool)]:
            b.text = "'True'" if b.value else "'False'"

        # Spark temp views can't have prefixes, but we can treat prefixed
        # table names as bare, if the prefix matches the default search path.
        for t in query.xpath("//Table"):
            if "." in t.name and hasattr(query, 'compare'):
                search_path = query.compare.search_path
                if len(search_path) > 0:
                    schema = search_path[0] # only use first schema in path
                    t.name = t.name.replace(f"{schema}.", "")

        return str(query)


class SparkNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]

    def identifier_match(self, query, meta):
        return self.strip_escapes(query).lower() == self.strip_escapes(meta).lower()
