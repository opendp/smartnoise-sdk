from calendar import c
from .base import DbFactory, DbDataset

class PostgresFactory(DbFactory):
    def __init__(self, engine="postgres", user=None, host=None, port=None, datasets=...):
        super().__init__(engine, user, host, port, datasets)
    def connect(self, dataset):
        host = self.host
        user = self.user
        port = self.port
        password = self.password
        dbname = self.datasets[dataset]['dbname'] if 'dbname' in self.datasets[dataset] else None
        table_name = self.datasets[dataset]['table'] if 'table' in self.datasets[dataset] else 'PUMS.PUMS'

        try:
            import psycopg2
            connection = psycopg2.connect(host=host, port=port, user=user, password=password, database=dbname)
            self.connections[dataset] = DbDataset(connection, table_name)
            print(f'Postgres: Connected {dataset} to {dbname} as {table_name}')
        except Exception as e:
            print(str(e))
            print(f"Unable to connect to postgres dataset {dataset}.  Ensure connection info is correct and psycopg2 is installed")
