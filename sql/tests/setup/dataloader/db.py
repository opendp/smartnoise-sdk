import os
import subprocess
import pandas as pd
import copy
import yaml
import random
import copy

from snsql.sql import PrivateReader
from snsql.metadata import Metadata
from snsql.metadata import Table, Float, String

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

pums_csv_path = os.path.join(root_url,"datasets", "PUMS.csv")
pums_pid_csv_path = os.path.join(root_url,"datasets", "PUMS_pid.csv")
pums_large_csv_path = os.path.join(root_url,"datasets", "PUMS_large.csv")
pums_dup_csv_path = os.path.join(root_url,"datasets", "PUMS_dup.csv")

pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_large_schema_path = os.path.join(root_url,"datasets", "PUMS_large.yaml")
pums_pid_schema_path = os.path.join(root_url,"datasets", "PUMS_pid.yaml")
pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_dup_schema_path = os.path.join(root_url,"datasets", "PUMS_dup.yaml")


def _download_file(url, local_file):
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve
    urlretrieve(url, local_file)

def find_ngrams(input_list, n):
    return input_list if n == 1 else list(zip(*[input_list[i:] for i in range(n)]))

def download_data_files():
    iris_schema_path = os.path.join(root_url,"datasets", "iris.yaml")
    if not os.path.exists(iris_schema_path):
        iris = Table("iris", "iris", [
                    Float("sepal length (cm)", 4, 8),
                    Float("sepal width (cm)", 2, 5),
                    Float("petal length (cm)", 1, 7),
                    Float("petal width (cm)", 0, 3)
        ], 150)
        schema = Metadata([iris], "csv")
        schema.to_file(iris_schema_path, "iris")

    if not os.path.exists(pums_csv_path) or not os.path.exists(pums_pid_csv_path) or not os.path.exists(pums_large_csv_path):
        pums_url = "https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics_1000/data.csv"
        pums_large_url = "https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics/data.csv"
        _download_file(pums_url, pums_csv_path)
        _download_file(pums_large_url, pums_large_csv_path)
        df = pd.read_csv(pums_csv_path)
        df_pid = df.assign(pid = [i for i in range(1, 1001)])
        df_pid.to_csv(pums_pid_csv_path, index=False)

    if not os.path.exists(pums_dup_csv_path):
        random.seed(1011)
        df_pid = pd.read_csv(pums_pid_csv_path)
        new_records = []
        for _ in range(2):
            for idx, row in df_pid.iterrows():
                if row['sex'] == 1.0:
                    p = 0.22
                else:
                    p = 0.56
                if random.random() < p:
                    new_records.append(row)
        for row in new_records:
            df_pid = df_pid.append(row)
        df_pid = df_pid.astype(int)
        df_pid.to_csv(pums_dup_csv_path, index=False)

    reddit_dataset_path = os.path.join(root_url,"datasets", "reddit.csv")
    if not os.path.exists(reddit_dataset_path):
        import re
        reddit_url = "https://github.com/joshua-oss/differentially-private-set-union/raw/master/data/clean_askreddit.csv.zip"
        reddit_zip_path = os.path.join(root_url,"datasets", "askreddit.csv.zip")
        datasets = os.path.join(root_url,"datasets")
        clean_reddit_path = os.path.join(datasets, "clean_askreddit.csv")
        _download_file(reddit_url, reddit_zip_path)
        from zipfile import ZipFile
        with ZipFile(reddit_zip_path) as zf:
            zf.extractall(datasets)
        reddit_df = pd.read_csv(clean_reddit_path, index_col=0)
        reddit_df = reddit_df.sample(frac=0.01)
        reddit_df['clean_text'] = reddit_df['clean_text'].astype(str)
        reddit_df.loc[:,'clean_text'] = reddit_df.clean_text.apply(lambda x : str.lower(x))
        reddit_df.loc[:,'clean_text'] = reddit_df.clean_text.apply(lambda x : " ".join(re.findall('[\w]+', x)))
        reddit_df['ngram'] = reddit_df['clean_text'].map(lambda x: find_ngrams(x.split(" "), 2))
        rows = list()
        for row in reddit_df[['author', 'ngram']].iterrows():
            r = row[1]
            for ngram in r.ngram:
                rows.append((r.author, ngram))
        ngrams = pd.DataFrame(rows, columns=['author', 'ngram'])
        ngrams.to_csv(reddit_dataset_path)


    reddit_schema_path = os.path.join(root_url,"datasets", "reddit.yaml")
    if not os.path.exists(reddit_schema_path):
        reddit = Table("reddit", "reddit",  [
                    String("author", card=10000, is_key=True),
                    String("ngram", card=10000)
        ], 500000, None, False, max_ids=500)
        schema = Metadata([reddit], "csv")
        schema.to_file(reddit_schema_path, "reddit")

class DbEngine:
    # Connections to a list of test databases sharing connection info
    def __init__(self, engine, user=None, host=None, port=None, databases={}):
        self.engine = engine
        self.user = user
        self.host = host
        self.port = port
        if databases == {}:
            if self.engine.lower() == "pandas":
                databases={'PUMS': 'PUMS', 'PUMS_pid': 'PUMS_pid', 'PUMS_dup': 'PUMS_dup'}
            elif self.engine.lower() == "spark":
                databases = {'PUMS': 'PUMS', 'PUMS_pid': 'PUMS_pid', 'PUMS_dup': 'PUMS_dup', 'PUMS_large': 'PUMS_large'}
        self.databases = databases
        env_passwd = engine.upper() + "_" + "PASSWORD"
        password = os.environ.get(env_passwd)
        if password is not None:
            self.password = password
        else:
            conn = f"{engine}://{host}:{port}"
            try:
                import keyring
                password = keyring.get_password(conn, user)
                self.password = password
            except:
                print(f"No password for engine {conn}")
                print("Please make sure password is set and keyring is installed")
                self.password = ""
        self.connections = {}
        for database in self.databases:
            self.connect(database)
    def connect(self, database):
        host = self.host
        user = self.user
        port = self.port
        engine = self.engine
        password = self.password
        if database not in self.databases:
            raise ValueError(f"Database {database} is not available for {engine}")
        dbname = self.databases[database]
        if self.engine.lower() == "postgres":
            try:
                import psycopg2
                self.connections[database] = psycopg2.connect(host=host, port=port, user=user, password=password, database=dbname)
                print(f'Postgres: Connected {database} to {dbname}')
            except:
                print(f"Unable to connect to postgres database {database}.  Ensure connection info is correct and psycopg2 is installed")
        elif self.engine.lower() == "pandas":
            self.connections['PUMS'] = pd.read_csv(pums_csv_path)
            self.connections['PUMS_pid'] = pd.read_csv(pums_pid_csv_path)
            self.connections['PUMS_dup'] = pd.read_csv(pums_dup_csv_path)
            print(f'Pandas: Connected to 3 databases')
        elif self.engine.lower() == "sqlserver":
            try:
                import pyodbc
                dsn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={host},{port};UID={user};Database={dbname};PWD={password}"
                self.connections[database] = pyodbc.connect(dsn)
                print(f'SQL Server: Connected {database} to {dbname}')
            except:
                print(f"Unable to connect to SQL Server database {database}.  Ensure connection info is correct and pyodbc is installed.")
        elif self.engine.lower() == "spark":
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.getOrCreate()
                pums_pid = spark.read.load(pums_pid_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                pums_pid.createOrReplaceTempView("PUMS") # use same table for PUMS and PUMS_pid
                pums_dup = spark.read.load(pums_dup_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                pums_dup.createOrReplaceTempView("PUMS_dup")
                pums_large = spark.read.load(pums_large_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                colnames = list(pums_large.columns)
                colnames[0] = "PersonID"
                pums_large = pums_large.toDF(*colnames)
                pums_large.createOrReplaceTempView("PUMS_large")
                self.connections['PUMS'] = spark
                self.connections['PUMS_pid'] = spark
                self.connections['PUMS_dup'] = spark
                self.connections['PUMS_large'] = spark
                print(f'Spark: Connected to 4 databases')
            except:
                print("Unable to connect to Spark test databases.  Make sure pyspark is installed.")
        else:
            print(f"Unable to connect to databases for engine {self.engine}")
    @property
    def dialect(self):
        engine = self.engine
        dialects = {
            'postgres': 'postgresql+psycopg2',
            'sqlserver': 'mssql+pyodbc',
            'mysql': 'mysql+pymysql'
        }
        return None if engine not in dialects else dialects[engine]
    def get_private_reader(self, *ignore, metadata, privacy, database, **kwargs):
        if database not in self.connections:
            return None
        else:
            conn = self.connections[database]
            priv = PrivateReader.from_connection(conn, metadata=metadata, privacy=privacy)
            if self.engine.lower() == "spark":
                priv.reader.compare.search_path = ["PUMS"]
            return priv
    def get_connection(self, *ignore, database, **kwargs):
        if database not in self.connections:
            return None
        else:
            return self.connections[database]

class DbCollection:
    # Collection of test databases keyed by engine and database name.
    # Automatically connects to databases listed in connections-unit.yaml
    def __init__(self):
        self.metadata = {
            'PUMS': Metadata.from_file(pums_schema_path),
            'PUMS_large': Metadata.from_file(pums_large_schema_path),
            'PUMS_pid': Metadata.from_file(pums_pid_schema_path),
            'PUMS_dup': Metadata.from_file(pums_dup_schema_path)
        }
        self.engines = {}
        home = os.path.expanduser("~")
        p = os.path.join(home, ".smartnoise", "connections-unit.yaml")
        if not os.environ.get('SKIP_PANDAS'):
            self.engines['pandas'] = DbEngine('pandas')
        else:
            print("Skipping pandas database tests")
        if os.environ.get('TEST_SPARK'):
            self.engines['spark'] = DbEngine('spark')
        else:
            print("TEST_SPARK not set, so skipping Spark tests")
        if not os.path.exists(p):
            print ("No config file at ~/.smartnoise/connections-unit.yaml")
        else:
            with open(p, 'r') as stream:
                conns = yaml.safe_load(stream)
            if conns is None:
                print("List of installed test engines is empty")
            else:
                for engine in conns:
                    eng = conns[engine]
                    host = conns[engine]["host"]
                    port = conns[engine]["port"]
                    user = conns[engine]["user"]
                    databases = eng['databases']
                    self.engines[engine] = DbEngine(engine, user, host, port, databases)
    def __str__(self):
        description = ""
        for engine in self.engines:
            eng = self.engines[engine]
            description += f"{eng.user}@{engine}://{eng.host}:{eng.port}\n"
            for database in eng.databases:
                dbdest = eng.databases[database]
                connected = "(connected)" if database in eng.connections else ""
                description += f"\t{database} -> {dbdest} {connected}\n"
        return description
    def get_private_readers(self, *ignore, metadata=None, privacy, database, engine=None, overrides={}, **kwargs):
        readers = []
        if metadata is None and database in self.metadata:
            metadata = self.metadata[database]
        if metadata is None:
            print(f"No metadata available for {database}")
            return []
        if isinstance(metadata, str):
            metadata = Metadata.from_file(metadata)
        if len(overrides) > 0:
            # make a copy
            metadata = copy.deepcopy(metadata)
        # apply overrides to only the first table in the metadata
        table_name = list(metadata.m_tables)[0]
        table = metadata.m_tables[table_name]
        for propname in overrides:
            propval = overrides[propname]
            if propname == 'censor_dims':
                table.censor_dims = propval
            elif propname == 'clamp_counts':
                table.clamp_counts = propval
            elif propname == 'max_ids' or propname == 'max_contrib':
                table.max_ids = propval
            else:
                print(f"Unable to set override for {propname}={propval}")
        if engine is not None:
            engines = [engine]
        else:
            engines = [eng for eng in self.engines]
        for engine in engines:
            if engine in self.engines:
                eng = self.engines[engine]
                reader = eng.get_private_reader(metadata=metadata, privacy=privacy, database=database)
                if reader is not None:
                    readers.append(reader)
        return readers
    def get_private_reader(self, *ignore, metadata=None, privacy, database, engine, overrides={}, **kwargs):
        readers = self.get_private_readers(metadata=metadata, privacy=privacy, database=database, engine=engine, overrides=overrides)
        return None if len(readers) == 0 else readers[0]
    def get_connection(self, *ignore, database, engine, **kwargs):
        if engine in self.engines:
            eng = self.engines[engine]
            return eng.get_connection(database=database)
        else:
            return None
    def get_dialect(self, *ignore, engine, **kwargs):
        if engine in self.engines:
            eng = self.engines[engine]
            return eng.dialect
        else:
            return None
    def to_tuples(self, rowset):
        if hasattr(rowset, 'toLocalIterator'): # it's RDD
            colnames = rowset.columns
            try:
                return [colnames] + [[c for c in r] for r in rowset.toLocalIterator()]
            except:
                return [colnames]
        else:
            return rowset
