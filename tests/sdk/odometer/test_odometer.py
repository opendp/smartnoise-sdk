import os
import subprocess
from opendp.smartnoise.sql.privacy import Privacy
from opendp.smartnoise.sql.private_reader import PrivateReader
import pandas as pd


from opendp.smartnoise.metadata.collection import CollectionMetadata

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))
pums = pd.read_csv(csv_path)
privacy = Privacy(epsilon=1.0)

meta_obj = CollectionMetadata.from_(meta_path)

class TestOdometer:
    def test_count_pid_query(self):
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_path)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(DISTINCT pid) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 1)
    def test_count_query(self):
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_path)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_count_row_privacy(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(*) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 1)
    def test_count_row_privacy_col(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 1)
    def test_count_row_privacy_col_quantifier(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(DISTINCT age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_count_row_privacy_no_censor(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = False
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        assert(priv.odometer.spent == (0.0, 0.0))
        assert(priv.odometer.k == 0)
        res = priv.execute("SELECT COUNT(DISTINCT age) FROM PUMS.PUMS GROUP BY educ")
        # This is a bug.  This should be 1, because keycount is never used
        assert(priv.odometer.k == 2)
    def test_variance(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT VAR(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 3)
    def test_std(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT STD(age) FROM PUMS.PUMS GROUP BY sex")
        assert(priv.odometer.k == 3)
    def test_avg(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT AVG(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_sum(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT SUM(age) FROM PUMS.PUMS GROUP BY educ")
        assert(priv.odometer.k == 2)
    def test_three_var(self):
        meta_obj['PUMS.PUMS'].row_privacy = True
        meta_obj['PUMS.PUMS']['pid'].is_key = False
        meta_obj['PUMS.PUMS'].censor_dims = True
        priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta_obj)
        res = priv.execute("SELECT VAR(age), VAR(educ), VAR(income) FROM PUMS.PUMS GROUP BY sex")
        assert(priv.odometer.k == 5)
