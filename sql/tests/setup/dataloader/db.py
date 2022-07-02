import os
import subprocess
import pandas as pd
import copy
import yaml
import copy

from .factories.bigquery import BigQueryFactory
from .factories.mysql import MySqlFactory
from .factories.pandas import PandasFactory
from .factories.postgres import PostgresFactory
from .factories.spark import SparkFactory
from .factories.sqlserver import SqlServerFactory

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_large_schema_path = os.path.join(root_url,"datasets", "PUMS_large.yaml")
pums_pid_schema_path = os.path.join(root_url,"datasets", "PUMS_pid.yaml")
pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_dup_schema_path = os.path.join(root_url,"datasets", "PUMS_dup.yaml")
pums_null_schema_path = os.path.join(root_url,"datasets", "PUMS_dup.yaml")


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
            self.engines['pandas'] = PandasFactory()
        else:
            print("Skipping pandas database tests")
        if os.environ.get('TEST_SPARK'):
            self.engines['spark'] = SparkFactory()
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
                    if 'databases' in eng:
                        raise ValueError(f"connections-unit.yaml has a 'databases' section for engine {engine}.  Please update to use 'datasets' syntax.")
                    datasets = eng['datasets']

                    if engine == "postgres":
                        self.engines[engine] = PostgresFactory(engine, user, host, port, datasets)
                    elif engine == "sqlserver":
                        self.engines[engine] = SqlServerFactory(engine, user, host, port, datasets)
                    elif engine == "mysql":
                        self.engines[engine] = MySqlFactory(engine, user, host, port, datasets)
                    elif engine == "bigquery":
                        self.engines[engine] = BigQueryFactory(engine, user, host, port, datasets)

    def __str__(self):
        description = ""
        for engine in self.engines:
            eng = self.engines[engine]
            description += f"{eng.user}@{engine}://{eng.host}:{eng.port}\n"
            for dataset in eng.datasets:
                dbdest = eng.datasets[dataset]
                connected = "(connected)" if dataset in eng.connections else ""
                description += f"\t{dataset} -> {dbdest} {connected}\n"
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
            elif propname == 'clamp_columns':
                table.clamp_columns = propval
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
                    reader = eng.get_private_reader(metadata=metadata, privacy=privacy, dataset=database)
                except Exception as e:
                    print(str(e))
                    raise ValueError(f"Unable to get a private reader for dataset {database} using {engine}")
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
            if hasattr(rowset, 'columns'):
                colnames = rowset.columns
                try:
                    return [colnames] + [[c for c in r] for r in rowset.toLocalIterator()]
                except:
                    return [colnames]
            else:
                return [[c for c in r] for r in rowset.collect()]
        else:
            return rowset

def download_data_files():
    from dataloader.download_reddit import download_reddit
    from dataloader.download_pums import download_pums
    from dataloader.make_sqlite import make_sqlite

    download_reddit()
    download_pums()
    make_sqlite()
