from snsql.sql.reader.base import SqlReader, NameCompare, Serializer
from snsql.sql.reader.pandas import PandasReader, PandasNameCompare, PandasSerializer
from snsql.sql.reader.postgres import PostgresReader, PostgresNameCompare, PostgresSerializer
from snsql.sql.reader.sql_server import SqlServerReader, SqlServerNameCompare, SqlServerSerializer
from snsql.sql.reader.spark import SparkReader, SparkNameCompare, SparkSerializer
from snsql.sql.reader.presto import PrestoReader, PrestoNameCompare, PrestoSerializer
from snsql.sql.reader.bigquery import BigQueryReader, BigQueryNameCompare, BigQuerySerializer

import os
import subprocess
import pandas as pd

from snsql.metadata import Metadata

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))


class TestFromConnWithEngine:
    # New preferred style where established connection is passed in
    def test_postgres(self):
        engine = "postgres"
        reader = SqlReader.from_connection(None, engine)
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, PostgresReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, PostgresNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, PostgresSerializer))
    def test_pandas(self):
        engine = "pandas"
        meta = Metadata.from_file(meta_path)
        df = pd.read_csv(csv_path)
        reader = SqlReader.from_connection(df, engine, metadata=meta)
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, PandasReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, PandasNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, PandasSerializer))
    def test_sqlserver(self):
        engine = "sqlserver"
        reader = SqlReader.from_connection(None, engine)
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, SqlServerReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, SqlServerNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, SqlServerSerializer))
    def test_spark(self):
        engine = "spark"
        reader = SqlReader.from_connection(None, engine)
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, SparkReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, SparkNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, SparkSerializer))
    def test_presto(self):
        engine = "presto"
        reader = SqlReader.from_connection(None, engine)
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, PrestoReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, PrestoNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, PrestoSerializer))

class TestWithoutConn:
    # test the legacy style where the reader creates the connection
    def test_postgres(self):
        reader = PostgresReader("localhost", "pums", "postgres", "foo")
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, PostgresReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, PostgresNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, PostgresSerializer))
    def test_pandas(self):
        engine = "pandas"
        meta = Metadata.from_file(meta_path)
        df = pd.read_csv(csv_path)
        reader = PandasReader(df, meta)
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, PandasReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, PandasNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, PandasSerializer))
    def test_sqlserver(self):
        reader = SqlServerReader("localhost", "PUMS", "sa", "foo")
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, SqlServerReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, SqlServerNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, SqlServerSerializer))
    def test_spark(self):
        engine = "spark"
        reader = SparkReader(None)
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, SparkReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, SparkNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, SparkSerializer))
    def test_presto(self):
        engine = "presto"
        reader = PrestoReader("localhost", "PUMS", "presto", "foo")
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, PrestoReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, PrestoNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, PrestoSerializer))
    def test_bigquery(self):
        engine = "bigquery"
        reader = BigQueryReader("credentials_path", "conn")
        assert(isinstance(reader, SqlReader))
        assert(isinstance(reader, BigQueryReader))
        assert(isinstance(reader.compare, NameCompare))
        assert(isinstance(reader.compare, BigQueryNameCompare))
        assert(isinstance(reader.serializer, Serializer))
        assert(isinstance(reader.serializer, BigQuerySerializer))
