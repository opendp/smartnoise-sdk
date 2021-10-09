import importlib

#from snsql.metadata import Metadata
from .base import SqlReader, NameCompare, Serializer
from .engine import Engine
import copy
import warnings
import re


class PandasReader(SqlReader):
    ENGINE = Engine.PANDAS

    def __init__(self, df=None, metadata=None, conn=None, **kwargs):
        super().__init__(self.ENGINE)
        if conn is not None:
            df = conn
        if metadata is None:
            raise ValueError("Load without metadata is not yet implemented")
        if df is None:
            raise ValueError("Pass in a Pandas dataframe")
        self.df = df

        # we can replace this when we remove
        # Metadata from pandas cleaning
        class_ = getattr(importlib.import_module("snsql.metadata"), "Metadata")
        metadata = class_.from_(metadata)

        self.metadata, self.original_column_names = self._sanitize_metadata(metadata)
        import sqlite3

        ver = [int(part) for part in sqlite3.sqlite_version.split(".")]
        if len(ver) == 3:
            # all historical versions of SQLite have 3 parts
            if (
                ver[0] < 3
                or (ver[0] == 3 and ver[1] < 2)
                or (ver[0] == 3 and ver[1] == 2 and ver[2] < 6)
            ):
                warnings.warn(
                    "This python environment has outdated sqlite version {0}.  PandasReader will fail on queries that use private_key.  Please upgrade to a newer Python environment (with sqlite >= 3.2.6), or ensure that you only use row_privacy.".format(
                        sqlite3.sqlite_version
                    ),
                    Warning,
                )

    def _sanitize_column_name(self, column_name):
        x = re.search(r".*[a-zA-Z0-9()_]", column_name)
        if x is None:
            raise Exception(
                "Unsupported column name {}. Column names must be alphanumeric or _, (, ).".format(
                    column_name
                )
            )
        column_name = column_name.replace(" ", "_").replace("(", "_0_").replace(")", "_1_")
        return column_name

    def _sanitize_metadata(self, metadata):
        metadata = copy.deepcopy(metadata)
        table_names = list(metadata.m_tables.keys())
        if len(table_names) > 1:
            raise Exception(
                "Only one table is supported for PandasReader. {} found.".format(len(table_names))
            )
        table_name = table_names[0]
        original_column_names = list(metadata.m_tables[table_name].m_columns)

        has_key = False
        for column_name in original_column_names:
            sanitized_column_name = self._sanitize_column_name(column_name)
            metadata.m_tables[table_name].m_columns[sanitized_column_name] = metadata.m_tables[
                table_name
            ].m_columns[column_name]
            metadata.m_tables[table_name].m_columns[
                sanitized_column_name
            ].name = sanitized_column_name
            has_key = (
                has_key or metadata.m_tables[table_name].m_columns[sanitized_column_name].is_key
            )
            self.df[sanitized_column_name] = self.df[column_name]
            if column_name != sanitized_column_name:
                del metadata.m_tables[table_name].m_columns[column_name]

        if not has_key:  # TODO handle this in metadata to avoid circ dep
            key = "primary_key"
            self.df[key] = range(len(self.df))

            from snsql.metadata import Int

            metadata.m_tables[table_name].m_columns[key] = Int(
                key, minval=0, maxval=len(self.df), is_key=True
            )
        return metadata, original_column_names

    def _sanitize_query(self, query):
        for column in self.original_column_names:
            sanitized_column = self._sanitize_column_name(column)
            for column_form in ["'{}'".format(column), '"{}"'.format(column), column]:
                query = query.replace(column_form, sanitized_column)
        return query

    def db_name(self):
        """
            Get the database associated with this connection
        """
        sql = "SELECT current_database();"
        dbname = self.execute(sql)[1][0]
        return dbname

    def execute(self, query, *ignore, accuracy:bool=False):
        """
            Executes a raw SQL string against the database and returns
            tuples for rows.  This will NOT fix the query to target the
            specific SQL dialect.  Call execute_typed to fix dialect.
        """
        query = self._sanitize_query(query)
        from pandasql import sqldf

        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        table_names = list(self.metadata.m_tables.keys())
        if len(table_names) > 1:
            raise Exception(
                "PandasReader only supports one table, {} found.".format(len(table_names))
            )

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
        return [tuple([col for col in q_result.columns])] + [
            val[1:] for val in q_result.itertuples()
        ]

class PandasNameCompare(NameCompare):
    def __init__(self, search_path=None):
        super().__init__(search_path)

class PandasSerializer(Serializer):
    def __init__(self):
        super().__init__()
