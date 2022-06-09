from .base import DbFactory, DbDataset
import os

class BigQueryFactory(DbFactory):
    def __init__(self, engine="mysql", user=None, host=None, port=None, datasets=...):
        super().__init__(engine, user, host, port, datasets)
    def connect(self, dataset):
        dbname = self.datasets[dataset]['dbname'] if 'dbname' in self.datasets[dataset] else None
        table_name = self.datasets[dataset]['table'] if 'table' in self.datasets[dataset] else 'PUMS.PUMS'
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            import json
            print("Running bigquery tests and looking for credentials.")
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                creds = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                info = json.loads(creds, strict=False)
                credentials = service_account.Credentials.from_service_account_info(
                    info
                )
                conn = bigquery.Client(credentials=credentials, project=credentials.project_id)
                self.connections[dataset] = DbDataset(conn, table_name)
                print(f'BigQuery: Connected {dataset} to {dbname} as {table_name}')
        except:
            print(f"Unable to connect to BigQuery dataset {dataset}.  Ensure connection info is correct and gcp library is installed.")
