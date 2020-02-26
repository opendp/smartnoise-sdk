import os
import re

from .base import Base, NameCompare
from burdock.sql.ast.tokens import Literal
from burdock.sql.ast.expressions.numeric import BareFunction

class SparkReader(Base):
    def __init__(self, host, session, user, password=None, port=None):
        from pyspark.sql import SparkSession
        self.api = session
        self.engine = "Spark"
        self.database = "Spark Session"
        self.serializer = SparkSerializer()
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
    def db_name(self):
        return self.database

    def update_connection_string(self):
        self.connection_string = None
        pass

class SparkSerializer:
    def serialize(self, query):
        for re in [n for n in query.find_nodes(BareFunction) if n.name == 'RANDOM']:
            re.name = 'rand'
        
        for b in [n for n in query.find_nodes(Literal) if isinstance(n.value, bool)]:
            b.text = "'True'" if b.value else "'False'"

        return(str(query))

class SparkNameCompare(NameCompare):
    def __init__(self, search_path=None):
        self.search_path = search_path if search_path is not None else ["dbo"]
    def identifier_match(self, query, meta):
        return self.strip_escapes(query).lower() == self.strip_escapes(meta).lower()
