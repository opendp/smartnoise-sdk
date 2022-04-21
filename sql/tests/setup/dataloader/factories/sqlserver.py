from .base import DbFactory, DbDataset

class SqlServerFactory(DbFactory):
    def __init__(self, engine="sqlserver", user=None, host=None, port=None, datasets=...):
        super().__init__(engine, user, host, port, datasets)
    def connect(self, dataset):
        dbname = self.datasets[dataset]['dbname'] if 'dbname' in self.datasets[dataset] else None
        table_name = self.datasets[dataset]['table'] if 'table' in self.datasets[dataset] else 'PUMS.PUMS'
        try:
            import pyodbc
            if self.host.startswith('(localdb)'):
                dsn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={self.host};Database={dbname}"
            else:
                dsn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={self.host},{self.port};UID={self.user};Database={dbname};PWD={self.password}"
            self.connections[dataset] = DbDataset(pyodbc.connect(dsn), table_name)
            print(f'SQL Server: Connected {dataset} to {dbname} as {table_name}')
        except:
            print(f"Unable to connect to SQL Server dataset {dataset}.  Ensure connection info is correct and pyodbc is installed.")
