import os
import subprocess
import sys
import time
from opendp.smartnoise.sql.privacy import Privacy
import sklearn.datasets
import pandas as pd
import keyring

from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.metadata.collection import Table, Float, String

from subprocess import Popen, PIPE
from threading import Thread

import pytest
import yaml

from requests import Session

from opendp.smartnoise.sql import PrivateReader, PandasReader, PostgresReader, SqlServerReader
from opendp.smartnoise.metadata.collection import CollectionMetadata

from opendp.smartnoise.client import _get_client
from opendp.smartnoise.client.restclient.rest_client import RestClient
from opendp.smartnoise.client.restclient.models.secret import Secret
DATAVERSE_TOKEN_ENV_VAR = "SMARTNOISE_DATAVERSE_TEST_TOKEN"

# Add the utils directory to the path
root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
sys.path.append(os.path.join(root_url, "utils"))

iris_dataset_path = os.path.join(root_url,"datasets", "iris.csv")
if not os.path.exists(iris_dataset_path):
    sklearn_dataset = sklearn.datasets.load_iris()
    sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    sklearn_df.to_csv(iris_dataset_path)


iris_schema_path = os.path.join(root_url,"datasets", "iris.yaml")
if not os.path.exists(iris_schema_path):
    iris = Table("iris", "iris", [
                Float("sepal length (cm)", 4, 8),
                Float("sepal width (cm)", 2, 5),
                Float("petal length (cm)", 1, 7),
                Float("petal width (cm)", 0, 3)
    ], 150)
    schema = CollectionMetadata([iris], "csv")
    schema.to_file(iris_schema_path, "iris")

def find_ngrams(input_list, n):
    return input_list if n == 1 else list(zip(*[input_list[i:] for i in range(n)]))

def _download_file(url, local_file):
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve
    urlretrieve(url, local_file)

pums_csv_path = os.path.join(root_url,"datasets", "PUMS.csv")
pums_pid_csv_path = os.path.join(root_url,"datasets", "PUMS_pid.csv")
if not os.path.exists(pums_csv_path) or not os.path.exists(pums_pid_csv_path):
    pums_url = "https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics_1000/data.csv"
    _download_file(pums_url, pums_csv_path)
    df = pd.read_csv(pums_csv_path)
    df_pid = df.assign(pid = [i for i in range(1, 1001)])
    df_pid.to_csv(pums_pid_csv_path)

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
    schema = CollectionMetadata([reddit], "csv")
    schema.to_file(reddit_schema_path, "reddit")

pums_schema_path = os.path.join(root_url,"datasets", "PUMS_row.yaml")
pums_large_schema_path = os.path.join(root_url,"datasets", "PUMS_large.yaml")
pums_pid_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")


@pytest.fixture(scope="session")
def client():
    client = _get_client()
    if DATAVERSE_TOKEN_ENV_VAR in os.environ:
        import pdb; pdb.set_trace()
        client.secretsput(Secret(name="dataverse:{}".format("demo_dataverse"),
                                 value=os.environ[DATAVERSE_TOKEN_ENV_VAR]))
    return client


class TestDbEngine:
    # Connections to a list of test databases sharing connection info
    def __init__(self, engine, user, host, port, databases):
        self.metadata = {
            'PUMS': pums_schema_path,
            'PUMS_large': pums_large_schema_path,
            'PUMS_pid': pums_pid_schema_path
        }
        self.engine = engine
        self.user = user
        self.host = host
        self.port = port
        self.databases = databases
        env_passwd = engine.upper() + "_" + "PASSWORD"
        password = os.environ.get(env_passwd)
        if password is not None:
            self.password = password
        else:
            conn = f"{engine}://{host}:{port}"
            password = keyring.get_password(conn, user)
            self.password = password
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
            except:
                print(f"Unable to connect to postgres database {database}.  Ensure connection info is correct and psycopg2 is installed")
        elif self.engine.lower() == "pandas":
            self.connections['PUMS'] = pd.read_csv(pums_csv_path)
            print(self.connections['PUMS'])
        elif self.engine.lower() == "sqlserver":
            try:
                import pyodbc
                dsn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={host},{port};UID={user};Database={dbname};PWD={password}"
                self.connections[database] = pyodbc.connect(dsn)
            except:
                print(f"Unable to connect to SQL Server database {database}.  Ensure connection info is correct and pyodbc is installed.")
    def create_private_reader(self, *ignore, metadata, privacy, database, **kwargs):
        if database not in self.connections:
            return None
        else:
            conn = self.connections[database]
            return PrivateReader.from_connection(conn, metadata=metadata, privacy=privacy)

class TestDbCollection:
    # Collection of test databases keyed by engine and database name.
    # Automatically connects to databases listed in connections-unit.yml
    def __init__(self):
        self.engines = {}
        home = os.path.expanduser("~")
        p = os.path.join(home, ".smartnoise", "connections-unit.yaml")
        if not os.environ.get('SKIP_PANDAS'):
            conns = TestDbEngine('pandas', None, None, None, {'PUMS': 'PUMS'})
            self.engines['pandas'] = conns
        else:
            print("Skipping pandas database tests")
        if os.environ.get('TEST_SPARK'):
            pass
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
                    self.engines[engine] = TestDbEngine(engine, user, host, port, databases)
    def __str__(self):
        description = ""
        for engine in self.engines:
            eng = self.engines[engine]
            description += f"{engine} - {eng.user}@{eng.host}:{eng.port}\n"
            #description += str(eng.databases)
            for database in eng.databases:
                dbdest = eng.databases[database]
                connected = "(connected)" if database in eng.connections else ""
                description += f"\t{database} -> {dbdest} {connected}\n"
        return description
    def create_private_readers(self, *ignore, metadata, privacy, database, engine=None, **kwargs):
        readers = []
        if engine is not None:
            engines = [engine]
        else:
            engines = [eng for eng in self.engines]
        for engine in engines:
            eng = self.engines[engine]
            reader = eng.create_private_reader(metadata=metadata, privacy=privacy, database=database)
            if reader is not None:
                readers.append(reader)
        return readers

dbcol = TestDbCollection()
print(dbcol)

@pytest.fixture(scope="module")
def test_databases():
    return dbcol


def test_client(client):
    pass
