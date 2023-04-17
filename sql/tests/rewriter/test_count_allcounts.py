import os
import subprocess
from snsql.metadata import Metadata
from snsql import from_df, Privacy
import pandas as pd

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
pums_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_null.csv"))

class TestCountAllCounts:
    def test_allcount_first(self):
        meta = Metadata.from_file(meta_path)
        privacy = Privacy(epsilon=10.0, delta=0.1)
        pums = pd.read_csv(pums_path)
        reader = from_df(pums, privacy=privacy, metadata=meta)
        query = 'SELECT COUNT(*) AS c, COUNT(sex) AS s, COUNT(age) AS a FROM PUMS.PUMS'
        result = reader.execute(query)
        counts = [int(v) for v in result[1]]
        assert not (counts[0] == counts[1] == counts[2])
    def test_allcount_last(self):
        meta = Metadata.from_file(meta_path)
        privacy = Privacy(epsilon=10.0, delta=0.1)
        pums = pd.read_csv(pums_path)
        reader = from_df(pums, privacy=privacy, metadata=meta)
        query = 'SELECT COUNT(sex) AS s, COUNT(age) AS a,  COUNT(*) AS c FROM PUMS.PUMS'
        result = reader.execute(query)
        counts = [int(v) for v in result[1]]
        assert not (counts[0] == counts[1] == counts[2])
