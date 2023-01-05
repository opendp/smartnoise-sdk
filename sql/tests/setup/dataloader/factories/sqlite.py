from .base import DbFactory, DbDataset
import sqlite3

class SQLiteFactory(DbFactory):
    def __init__(self, engine="sqlite", user=None, host=None, port=None, 
        datasets={'PUMS': 'PUMS', 'PUMS_pid': 'PUMS_pid', 'PUMS_dup': 'PUMS_dup', 'PUMS_null' : 'PUMS_null'}):
        super().__init__(engine, user, host, port, datasets)

    def connect(self, dataset):
        csv_paths = {
            'PUMS': self.pums_csv_path.replace('.csv', '.db'),
            'PUMS_pid': self.pums_pid_csv_path.replace('.csv', '.db'),
            'PUMS_dup': self.pums_dup_csv_path.replace('.csv', '.db'),
            'PUMS_null': self.pums_null_csv_path.replace('.csv', '.db')
        }

        if dataset in csv_paths:
            recordset = sqlite3.connect(csv_paths[dataset])
            self.connections[dataset] = DbDataset(recordset, 'PUMS')
        else:
            raise ValueError(f"We don't know how to connect to dataset {dataset} in Pandas")
