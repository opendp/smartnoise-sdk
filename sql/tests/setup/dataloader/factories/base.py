import os
import subprocess

class DbDataset:
    def __init__(self, connection, table_name):
        self.connection = connection
        self.table_name = table_name

class DbFactory:
    # Connections to a list of test databases sharing connection info
    def __init__(self, engine, user=None, host=None, port=None, datasets={}):
        self.engine = engine
        self.user = user
        self.host = host
        self.port = port
        self.datasets = datasets
        self.connections = {}

        root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
        self.pums_csv_path = os.path.join(root_url,"datasets", "PUMS.csv")
        self.pums_pid_csv_path = os.path.join(root_url,"datasets", "PUMS_pid.csv")
        self.pums_large_csv_path = os.path.join(root_url,"datasets", "PUMS_large.csv")
        self.pums_dup_csv_path = os.path.join(root_url,"datasets", "PUMS_dup.csv")
        self.pums_null_csv_path = os.path.join(root_url,"datasets", "PUMS_null.csv")

        # Get any passwords associated with this engine
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
        for dataset in datasets:
            self.connect(dataset)

    def connect(self, dataset):
        raise NotImplementedError("Not implemented on base class.  Please call a concrete implementation")

    @property
    def dialect(self):
        engine = self.engine
        dialects = {
            'postgres': 'postgresql+psycopg2',
            'sqlserver': 'mssql+pyodbc',
            'mysql': 'mysql+pymysql'
        }
        return None if engine not in dialects else dialects[engine]
    def get_private_reader(self, *ignore, metadata, privacy, dataset, **kwargs):
        if dataset not in self.connections:
            return None
        else:
            from .fixture_private_reader import FixturePrivateReader
            dbdataset = self.connections[dataset]
            conn = dbdataset.connection
            table_name = dbdataset.table_name
            if self.engine.lower() == "spark":
                conn = self.session
            priv = FixturePrivateReader.from_connection(
                conn, 
                metadata=metadata, 
                privacy=privacy,
                table_name=table_name
            )
            return priv
    def get_connection(self, *ignore, dataset, **kwargs):
        if dataset not in self.connections:
            return None
        else:
            return self.connections[dataset]