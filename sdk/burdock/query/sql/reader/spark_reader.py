import os
import re

from burdock.metadata.name_compare import BaseNameCompare
from .rowset import TypedRowset

from burdock.query.sql.ast.tokens import Literal
from burdock.query.sql.ast.expressions.numeric import BareFunction

class SparkReader:
    def __init__(self, host, database, user, password=None, port=None):
        from pyspark.sql import SparkSession
        self.api = SparkSession.builder.config("spark.rpc.askTimeout", "600s").config("spark.executor.memory", "6g").config("spark.driver.memory", "6g").config("spark.executor.memoryOverhead", "600").config("spark.driver.memoryOverhead", "600").getOrCreate()
        self.engine = "Spark"
        self.database = database
        self.serializer = SparkSQLSerializer()
        self.compare = SparkNameCompare()  
        self.update_connection_string()      

    def execute(self, query):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        
        df = self.api.sql(query)

        if df.columns is None:
            return []
        else:
            col_names = [tuple(c.lower() for c in df.columns)]
            import pyspark
            body = [c for c in df.collect() if type(c) is pyspark.sql.types.Row and c is not None] 
            return col_names + body

    """
        Executes a parsed AST and returns a typed recordset.
        Will fix to target approprate dialect. Needs symbols.
    """
    def execute_typed(self, query):
        if isinstance(query, str):
            raise ValueError("Please pass ASTs to execute_typed.  To execute strings, use execute.")

        syms = query.all_symbols()
        types = [s[1].type() for s in syms]
        sens = [s[1].sensitivity() for s in syms]

        if hasattr(self, 'serializer') and self.serializer is not None:
            query_string = self.serializer.serialize(query)
        else:
            query_string = str(query)
        rows = self.execute(query_string)
        return TypedRowset(rows, types, sens)

    def db_name(self):
        return self.database

    def update_connection_string(self):
        self.connection_string = None
        pass

class SparkSQLSerializer:
    def serialize(self, query):
        for re in [n for n in query.find_nodes(BareFunction) if n.name == 'RANDOM']:
            re.name = 'rand'
        
        for b in [n for n in query.find_nodes(Literal) if isinstance(n.value, bool)]:
            b.text = "'True'" if b.value else "'False'"

        return(str(query))

class SparkNameCompare(BaseNameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]
    def identifier_match(self, query, meta):
        return self.strip_escapes(query).lower() == self.strip_escapes(meta).lower()
