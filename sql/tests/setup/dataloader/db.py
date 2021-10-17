import os
import subprocess
import pandas as pd
import copy
import yaml
import copy

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

pums_csv_path = os.path.join(root_url,"datasets", "PUMS.csv")
pums_pid_csv_path = os.path.join(root_url,"datasets", "PUMS_pid.csv")
pums_large_csv_path = os.path.join(root_url,"datasets", "PUMS_large.csv")
pums_dup_csv_path = os.path.join(root_url,"datasets", "PUMS_dup.csv")
pums_null_csv_path = os.path.join(root_url,"datasets", "PUMS_null.csv")

pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_large_schema_path = os.path.join(root_url,"datasets", "PUMS_large.yaml")
pums_pid_schema_path = os.path.join(root_url,"datasets", "PUMS_pid.yaml")
pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_dup_schema_path = os.path.join(root_url,"datasets", "PUMS_dup.yaml")
pums_null_schema_path = os.path.join(root_url,"datasets", "PUMS_dup.yaml")

class DbEngine:
    # Connections to a list of test databases sharing connection info
    def __init__(self, engine, user=None, host=None, port=None, databases={}):
        self.engine = engine
        self.user = user
        self.host = host
        self.port = port
        if databases == {}:
            if self.engine.lower() == "pandas":
                databases={'PUMS': 'PUMS', 'PUMS_pid': 'PUMS_pid', 'PUMS_dup': 'PUMS_dup', 'PUMS_null' : 'PUMS_null'}
            elif self.engine.lower() == "spark":
                databases = {'PUMS': 'PUMS', 'PUMS_pid': 'PUMS_pid', 'PUMS_dup': 'PUMS_dup', 'PUMS_large': 'PUMS_large', 'PUMS_null' : 'PUMS_null'}
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
                print(f"Connecting to {conn} with no password")
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
            except Exception as e:
                print(str(e))
                print(f"Unable to connect to postgres database {database}.  Ensure connection info is correct and psycopg2 is installed")
        elif self.engine.lower() == "pandas":
            if 'PUMS' not in self.connections:
                self.connections['PUMS'] = pd.read_csv(pums_csv_path)
            if 'PUMS_pid' not in self.connections:
                self.connections['PUMS_pid'] = pd.read_csv(pums_pid_csv_path)
            if 'PUMS_dup' not in self.connections:
                self.connections['PUMS_dup'] = pd.read_csv(pums_dup_csv_path)
            if 'PUMS_null' not in self.connections:
                self.connections['PUMS_null'] = pd.read_csv(pums_null_csv_path)
        elif self.engine.lower() == "sqlserver":
            try:
                import pyodbc
                if host.startswith('(localdb)'):
                    dsn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={host};Database={dbname}"
                else:
                    dsn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={host},{port};UID={user};Database={dbname};PWD={password}"
                self.connections[database] = pyodbc.connect(dsn)
                print(f'SQL Server: Connected {database} to {dbname}')
            except:
                print(f"Unable to connect to SQL Server database {database}.  Ensure connection info is correct and pyodbc is installed.")
        elif self.engine.lower() == "mysql":
            try:
                import pymysql
                self.connections[database] = pymysql.connect(
                    host = host,
                    password = password,
                    user = user,
                    database = dbname
                )
            except:
                print(f"Unable to connect to MySQL database {database}.  Ensure connection info is correct and pymysql is installed.")
        elif self.engine.lower() == "sqlite":
            pass
        elif self.engine.lower() == "spark":
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.getOrCreate()
                self.session = spark
                if 'PUMS' not in self.connections:
                    pums = spark.read.load(pums_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                    self.connections['PUMS'] = pums
                if 'PUMS_pid' not in self.connections:
                    pums_pid = spark.read.load(pums_pid_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                    self.connections['PUMS_pid'] = pums_pid
                if 'PUMS_dup' not in self.connections:
                    pums_dup = spark.read.load(pums_dup_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                    self.connections['PUMS_dup'] = pums_dup
                if 'PUMS_null' not in self.connections:
                    pums_null = spark.read.load(pums_null_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                    self.connections['PUMS_null'] = pums_null
                if 'PUMS_large' not in self.connections:
                    pums_large = spark.read.load(pums_large_csv_path, format="csv", sep=",",inferSchema="true", header="true")
                    colnames = list(pums_large.columns)
                    colnames[0] = "PersonID"
                    pums_large = pums_large.toDF(*colnames)
                    self.connections['PUMS_large'] = pums_large
                    pums_large.createOrReplaceTempView("PUMS_large")
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
            from snsql.sql import PrivateReader
            conn = self.connections[database]
            if self.engine.lower() == "spark":
                if database.lower() != 'pums_large':
                    conn.createOrReplaceTempView("PUMS")
                conn = self.session
            priv = PrivateReader.from_connection(
                conn, 
                metadata=metadata, 
                privacy=privacy
            )
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
        from snsql.metadata import Metadata
        self.metadata = {
            'PUMS': Metadata.from_file(pums_schema_path),
            'PUMS_large': Metadata.from_file(pums_large_schema_path),
            'PUMS_pid': Metadata.from_file(pums_pid_schema_path),
            'PUMS_dup': Metadata.from_file(pums_dup_schema_path),
            'PUMS_null': Metadata.from_file(pums_dup_schema_path)
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
            from snsql.metadata import Metadata
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
                reader = None
                try:
                    reader = eng.get_private_reader(metadata=metadata, privacy=privacy, database=database)
                except:
                    pass
                finally:
                    if reader:
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

def download_data_files():
    from dataloader.download_reddit import download_reddit
    from dataloader.download_pums import download_pums
    from dataloader.make_sqlite import make_sqlite

    download_reddit()
    download_pums()
    make_sqlite()
