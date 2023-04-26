import os
import subprocess
import pandas as pd
from snsql import Privacy, from_connection
from snsql.sql.reader.pandas import PandasReader

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

pums_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))
pums_null_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_null.csv"))
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_two_table.yaml"))
privacy = Privacy(epsilon=30.0, delta=0.1)

class TestDictCounts:
    def test_pandas_no_dp(self):
        df_dict = {
            'pums': pd.read_csv(pums_path),
            'pums_null': pd.read_csv(pums_null_path)
        }
        reader = PandasReader(df_dict)
        rs = reader.execute("SELECT COUNT(*) FROM pums")
        assert rs[1][0] == 1000
        rs = reader.execute("SELECT COUNT(*) FROM pums_null")
        assert rs[1][0] > 1200
    def test_pandas_dict_from_connection(self):
        df_dict = {
            'PUMS.PUMS': pd.read_csv(pums_path),
            'PUMS.PUMS_null': pd.read_csv(pums_null_path)
        }
        reader = from_connection(df_dict, privacy=privacy, metadata=meta_path, engine='pandas')
        rs = reader.execute("SELECT COUNT(*) FROM PUMS.PUMS")
        assert rs[1][0] > 900 and rs[1][0] < 1100
        rs = reader.execute("SELECT COUNT(*) FROM PUMS.PUMS_null")
        assert rs[1][0] > 1200
    def test_pandas_dict_probe(self):
        df_dict = {
            'PUMS.PUMS': pd.read_csv(pums_path),
            'PUMS.PUMS_null': pd.read_csv(pums_null_path)
        }
        reader = from_connection(df_dict, privacy=privacy, metadata=meta_path)
        rs = reader.execute("SELECT COUNT(*) FROM PUMS.PUMS")
        assert rs[1][0] > 900 and rs[1][0] < 1100
        rs = reader.execute("SELECT COUNT(*) FROM PUMS.PUMS_null")
        assert rs[1][0] > 1200


