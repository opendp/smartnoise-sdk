import os
import pytest
import subprocess
import pandas as pd
from opendp.smartnoise.sql import PrivateReader, PandasReader, PostgresReader, SqlServerReader
from opendp.smartnoise.metadata.collection import CollectionMetadata

@pytest.fixture(scope="module")
def reader_factory():
    class Factory:
        def __init__(self):
            pass
        def create(self):
            raise ValueError("Must override create()")
        def create_private(self, meta, epsilon, delta):
            r = self.create()
            return PrivateReader(r, meta, epsilon, delta)
    class Pandas(Factory):
        def __init__(self):
            git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
            meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
            csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))
            self.df = pd.read_csv(csv_path)
            self.meta = CollectionMetadata.from_file(meta_path)
            self.installed = True
        def create(self):
            return PandasReader(self.df, self.meta)
    class Postgres(Factory):
        def __init__(self):
            self.password = os.environ.get('POSTGRES_PASSWORD')
            self.user = os.environ.get('POSTGRES_USER')
            if self.user is None:
                self.user = "postgres"
            self.host = os.environ.get('POSTGRES_HOST')
            if self.host is None:
                self.host = "localhost"
            self.port = os.environ.get('POSTGRES_PORT')
            if self.port is None:
                self.port = 5432
            self.db = os.environ.get('POSTGRES_DB')
            if self.db is None:
                self.db = "pums" # needs to be lowercase
            if self.password is None:
                self.installed = False
            else:
                self.installed = True
        def create(self):
            return PostgresReader(self.host, self.db, self.user, self.password, self.port)
    
    factories = [Pandas(), Postgres()]
    return [factory for factory in factories if factory.installed]