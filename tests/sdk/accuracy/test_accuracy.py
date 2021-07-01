import os
import subprocess
from opendp.smartnoise.sql._mechanisms.accuracy import Accuracy
import numpy as np
import math

import pandas as pd

from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.sql import PrivateReader, SqlReader
from opendp.smartnoise.sql.privacy import Privacy
from opendp.smartnoise.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

meta = CollectionMetadata.from_file(meta_path)
pums = pd.read_csv(csv_path)
query = 'SELECT AVG(age), STD(age), VAR(age), SUM(age), COUNT(age) FROM PUMS.PUMS'
q = QueryParser(meta).query(query)
reader = SqlReader.from_connection(pums, "pandas", metadata=meta)
priv = PrivateReader(reader, meta, 1.0)
subquery, root = priv.rewrite(query)

meta = CollectionMetadata.from_file('datasets/PUMS.yaml')


privacy = Privacy(alphas=[0.01, 0.05], delta=1/(math.sqrt(100) * 100))
acc = Accuracy(root, subquery, privacy)


class TestAccuracy:
    def test_count_sigma(self):
        sigma = acc.scale(sensitivity=1)
        assert(np.isclose(sigma, 3.8469))
    def test_count_accuracy(self):
        error = acc.count(alpha=0.05)
        assert(np.isclose(error, 7.53978))
        error_wide = acc.count(alpha=0.01)
        assert(np.isclose(error_wide, 9.909))
        assert(error_wide > error)
    def test_count_accuracy_small_delta(self):
        acc = Accuracy(root, subquery, privacy=Privacy(epsilon=1.0, delta=0.1))
        error = acc.count(alpha=0.01)
        error_wide = acc.count(alpha=0.05)
        assert(error_wide < error)
    def test_count_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'count']
        a = acc.count(alpha=0.01, properties=p[0], row=None)
        assert(np.isclose(a, 9.90895))
        a = acc.count(alpha=0.05, properties=p[0], row=None)
        assert(np.isclose(a, 7.5398))
    def test_sum_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'sum']
        a = acc.sum(alpha=0.01, properties=p[0], row=None)
        assert(np.isclose(a, 990.895))
        a = acc.sum(alpha=0.05, properties=p[0], row=None)
        assert(np.isclose(a, 753.978))
    def test_mean_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'mean']
        a = acc.mean(alpha=0.05, properties=p[0], row=(0, 100, 100 * 50, 30, 22))
        assert(np.isclose(a, 25.3707))
        a = acc.mean(alpha=0.01, properties=p[0], row=(0, 100, 100 * 50, 30, 22))
        assert(np.isclose(a, 30.1797))
