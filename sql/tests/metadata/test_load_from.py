import os
import subprocess
from snsql.sql.privacy import Privacy
from snsql.sql.private_reader import PrivateReader
from snsql.sql.reader.pandas import PandasReader
import pandas as pd


from snsql.metadata import Metadata

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))
pums = pd.read_csv(csv_path)
privacy = Privacy(epsilon=1.0)

class TestMetadataLoadFrom:
    def test_load_from_path(self):
        meta = Metadata.from_(meta_path)
        p = meta['PUMS.PUMS']
    def test_load_from_private_reader_path_connect(self):
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_path)
        assert(isinstance(priv.reader, PandasReader))
        res = priv.execute("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY sex")
        assert(len(res) == 3)
    def test_load_from_private_reader_path(self):
        reader = PandasReader(pums, meta_path)
        priv = PrivateReader(reader, meta_path, privacy=Privacy(epsilon=1.0))
        res = priv.execute("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY sex")
        assert(len(res) == 3)