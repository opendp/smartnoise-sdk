from .sql_base import SqlReader, NameCompare
from .engine import Engine

from opendp.whitenoise.ast.tokens import Literal
from opendp.whitenoise.ast.expressions.numeric import BareFunction


class SparkReader(SqlReader):
    ENGINE = Engine.SPARK

    def __init__(self, session):
        super().__init__(SparkNameCompare(), SparkSerializer())
        from pyspark.sql import SparkSession  # TODO how do we deal with reader dependencies
        self.api = session
        self.database = "Spark Session"

    def execute(self, query):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")

        df = self.api.sql(query)
        return df

    def execute_typed(self, query):
        return self.execute(query)

    def db_name(self):
        return self.database


class SparkSerializer:
    def serialize(self, query):
        for r_e in [n for n in query.find_nodes(BareFunction) if n.name == 'RANDOM']:
            r_e.name = 'rand'

        for b in [n for n in query.find_nodes(Literal) if isinstance(n.value, bool)]:
            b.text = "'True'" if b.value else "'False'"

        return(str(query))


class SparkNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]

    def identifier_match(self, query, meta):
        return self.strip_escapes(query).lower() == self.strip_escapes(meta).lower()
