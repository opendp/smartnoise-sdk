from .rowset import TypedRowset
from burdock.query.sql.metadata.name_compare import BaseNameCompare


class CSVReader:
    def __init__(self, metadata, df):
        self.metadata = metadata
        self.df = df
    """
        Get the database associated with this connection
    """
    def db_name(self):
        sql = "SELECT current_database();"
        dbname = self.execute(sql)[1][0]
        return dbname
    """
        Executes a raw SQL string against the database and returns
        tuples for rows.  This will NOT fix the query to target the
        specific SQL dialect.  Call execute_typed to fix dialect.
    """
    def execute(self, query):
        from pandasql import sqldf
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        df_for_diffpriv1234 = self.df
        df_name = "df_for_diffpriv1234"

        def clean_query(query):
            table_name = self.metadata.tables()[0].name
            return query.replace(".".join([table_name] * 2), df_name)

        q_result = sqldf(clean_query(query), locals())
        return [tuple([col for col in q_result.columns])] + [val[1:] for val in q_result.itertuples()]

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

