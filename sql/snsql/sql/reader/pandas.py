import importlib
import pandas as pd
from sqlalchemy import create_engine, text
from .base import SqlReader, NameCompare, Serializer
from .engine import Engine
import copy
import warnings
import re


class PandasReader(SqlReader):
    ENGINE = Engine.PANDAS

    def __init__(self, df=None, metadata=None, conn=None, *ignore, table_name=None, **kwargs):
        super().__init__(self.ENGINE)
        if metadata is not None:
            self.metadata = metadata
            class_ = getattr(importlib.import_module("snsql.metadata"), "Metadata")
            self.metadata = class_.from_(metadata)
        if conn is not None:
            df = conn
        if df is None:
            raise ValueError("Pass in a Pandas dataframe")
        table_dict = {}
        if isinstance(df, pd.DataFrame):        
            if table_name is None:
                if metadata is None:
                    raise ValueError("Must pass in table_name if metadata is not provided")
                table_names = list(self.metadata.m_tables.keys())
                if len(table_names) > 1:
                    raise ValueError(
                        "Must pass in table_name if metadata has more than one table"
                    )
                table_name = table_names[0]
            else:
                if metadata is not None:
                    table_names = list(metadata.m_tables.keys())
                    if table_name not in table_names:
                        raise ValueError(
                            "table_name {} not in metadata".format(table_name)
                        )
            table_dict[table_name] = df
        elif not isinstance(df, dict):
            raise ValueError("df must be a Pandas dataframe or a dictionary of dataframes")
        else:
            table_dict = df

        # check minimum version of sqlite
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

        # load all dataframes into a single sqlite database
        table_names = list(table_dict.keys())
        self.table_names = table_names

        table_name_fixed = [t.replace(".", "_") for t in table_names]
        if len(table_names) != len(set(table_name_fixed)):
            raise ValueError("Table names must be unique after replacing '.' with '_'.")
        db_engine = create_engine('sqlite://', echo=False)
        for table_name, df in table_dict.items():
            df.to_sql(table_name.replace('.','_'), con=db_engine, index=False)
        self.db_engine = db_engine

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
                key, lower=0, upper=len(self.df), is_key=True
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
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")
        
        # this is a hack; should use AST
        for table_name in self.table_names:
            query = query.replace(table_name, table_name.replace('.', '_'))
        with self.db_engine.connect() as conn:
            result_proxy = conn.execute(text(query))
            column_names = [tuple(c for c in result_proxy.keys())]
            result = [tuple(row) for row in result_proxy.fetchall()]
            return column_names + result

class PandasNameCompare(NameCompare):
    def __init__(self, search_path=None):
        super().__init__(search_path)

class PandasSerializer(Serializer):
    def __init__(self):
        super().__init__()
    def serialize(self, query):
        if isinstance(query, str):
            raise ValueError("We need an AST to validate and serialize.")
        return str(query)
