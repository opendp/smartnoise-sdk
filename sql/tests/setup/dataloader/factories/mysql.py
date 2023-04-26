from .base import DbFactory, DbDataset

class MySqlFactory(DbFactory):
    def __init__(self, engine="mysql", user=None, host=None, port=None, datasets=...):
        super().__init__(engine, user, host, port, datasets)
    def connect(self, dataset):
        dbname = self.datasets[dataset]['dbname'] if 'dbname' in self.datasets[dataset] else None
        table_name = self.datasets[dataset]['table'] if 'table' in self.datasets[dataset] else 'PUMS.PUMS'
        try:
            import pymysql
            self.connections[dataset] = DbDataset(
                pymysql.connect(
                    host = self.host,
                    password = self.password,
                    user = self.user,
                    database = dbname
                ),
                table_name
            )
            print(f'MySQL: Connected {dataset} to {dbname} as {table_name}')
        except Exception as e:
            print(f"Unable to connect to MySQL database {dataset}.  Ensure connection info is correct and pymysql is installed.")
            print(e)
