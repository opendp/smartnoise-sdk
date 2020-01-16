from .rowset import TypedRowset
from burdock.query.sql.metadata.name_compare import BaseNameCompare
from burdock.query.sql.metadata.metadata import Int
import copy
import re


class DataFrameReader:
    def __init__(self, metadata, df):
        self.original_column_names = []
        self.df = df
        self.metadata = self._sanitize_metadata(metadata)

    def _sanitize_column_name(self, column_name):
        x = re.search(r".*[a-zA-Z0-9()_]", column_name)
        if x is None:
            raise Exception("Unsupported column name {}. Column names must be alphanumeric or _, (, ).".format(
                column_name))
        column_name = column_name.replace(" ", "_").replace("(", "_0_").replace(")", "_1_")
        return column_name

    def _sanitize_metadata(self, metadata):
        metadata = copy.deepcopy(metadata)
        table_names = list(metadata.m_tables.keys())
        if len(table_names) > 1:
            raise Exception("Only one table is supported for DataFrameReader. {} found.".format(len(table_names)))
        table_name = table_names[0]
        self.original_column_names = list(metadata.m_tables[table_name].m_columns)

        has_key = False
        for column_name in self.original_column_names:
            sanitized_column_name = self._sanitize_column_name(column_name)
            metadata.m_tables[table_name].m_columns[sanitized_column_name] = metadata.m_tables[table_name].m_columns[column_name]
            metadata.m_tables[table_name].m_columns[sanitized_column_name].name = sanitized_column_name
            has_key = has_key or metadata.m_tables[table_name].m_columns[sanitized_column_name].is_key
            self.df[sanitized_column_name] = self.df[column_name]
            if column_name != sanitized_column_name:
                del metadata.m_tables[table_name].m_columns[column_name]

        if not has_key:
            key = "primary_key"
            self.df[key] = range(len(self.df))
            metadata.m_tables[table_name].m_columns[key] = Int(key,
            minval=0,
            maxval=len(self.df),
            is_key=True)
        return metadata

    def _sanitize_query(self, query):
        for column in self.original_column_names:
            sanitized_column = self._sanitize_column_name(column)
            for column_form in ["'{}'".format(column), '"{}"'.format(column), column]:
                query = query.replace(column_form, sanitized_column)
        return query
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
        query = self._sanitize_query(query)
        from pandasql import sqldf
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        table_names = list(self.metadata.m_tables.keys())
        if len(table_names) > 1:
            raise Exception("DataFrameReader only supports one table, {} found.".format(len(table_names)))

        df_name = "df_for_diffpriv1234"
        table_name = table_names[0]
        def clean_query(query):
            for column in self.metadata.m_tables[table_name].m_columns:
                if " " in column or "(" in column or ")" in column:
                    new_column_name = column.replace(" ", "_").replace("(", "_").replace(")", "_")
                    query = query.replace(column, new_column_name)
                    query = query.replace("'{}'".format(new_column_name), new_column_name)
            return query.replace(table_name, df_name)

        for column in self.metadata.m_tables[table_name].m_columns:
            new_column_name = column.replace(" ", "_").replace("(", "_").replace(")", "_")
            if self.metadata.m_tables[table_name].m_columns[column].is_key:
                if column not in self.df:
                    self.df[column] = range(len(self.df))
            else:
                if new_column_name not in self.df:
                    self.df[new_column_name] = self.df[column]

        df_for_diffpriv1234 = self.df
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
