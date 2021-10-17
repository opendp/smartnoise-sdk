import os
import subprocess
from snsql.sql._mechanisms.accuracy import Accuracy
import numpy as np
import math

import pandas as pd

from snsql.metadata import Metadata
from snsql.sql import PrivateReader
from snsql.sql.privacy import Privacy
from snsql.sql.parse import QueryParser

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

meta = Metadata.from_file(meta_path)
pums = pd.read_csv(csv_path)
query = 'SELECT AVG(age), STD(age), VAR(age), SUM(age), COUNT(age) FROM PUMS.PUMS GROUP BY sex'
q = QueryParser(meta).query(query)

privacy = Privacy(alphas=[0.01, 0.05], delta=1/(math.sqrt(100) * 100))
priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta)
subquery, root = priv._rewrite(query)

acc = Accuracy(root, subquery, privacy)

class TestAccuracy:
    def test_count_accuracy(self):
        error = acc.count(alpha=0.05)
        assert(error < 7.53978 and error > 0.5)
        error_wide = acc.count(alpha=0.01)
        assert(error_wide <9.909)
        assert(error_wide > error)
    def test_count_accuracy_small_delta(self):
        acc = Accuracy(root, subquery, privacy=Privacy(epsilon=1.0, delta=0.1))
        error = acc.count(alpha=0.01)
        error_wide = acc.count(alpha=0.05)
        assert(error_wide < error)
    def test_count_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'count']
        a = acc.count(alpha=0.01, properties=p[0], row=None)
        assert(a < 9.90895 and a > 0.0)
        a = acc.count(alpha=0.05, properties=p[0], row=None)
        assert(a and a < 7.5398)
    def test_sum_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'sum']
        a = acc.sum(alpha=0.01, properties=p[0], row=None)
        assert(a > 100.0 and a < 990.895)
        a = acc.sum(alpha=0.05, properties=p[0], row=None)
        assert(a > 100.0 and a < 753.978)
    def test_mean_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'mean']
        p = p[0]
        row = [0, 0, 0, 0, 0]
        sum_idx = p['columns']['sum']
        count_idx = p['columns']['count']
        row[sum_idx] = 100 * 50
        row[count_idx] = 100
        a = acc.mean(alpha=0.05, properties=p, row=row)
        assert(a > 10.0 and a < 25.3707)
        a = acc.mean(alpha=0.01, properties=p, row=row)
        assert(a > 10.0 and a < 30.1797)
    def test_var_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'variance']
        p = p[0]
        row = [0, 0, 0, 0, 0]
        sum_idx = p['columns']['sum']
        sum_s_idx = p['columns']['sum_of_squares']
        count_idx = p['columns']['count']
        row[sum_idx] = 100 * 51
        row[sum_s_idx] = 100 * (49 * 49)
        row[count_idx] = 100
        a = acc.variance(alpha=0.05, properties=p, row=row)
        assert(a > 1000.0 and a < 5509.55)
        a = acc.variance(alpha=0.01, properties=p, row=row)
        assert(a > 1000.0 and a < 6634.06837)
    def test_std_acc(self):
        p = [p for p in acc.properties if p and p['statistic'] == 'stddev']
        p = p[0]
        row = [0, 0, 0, 0, 0]
        sum_idx = p['columns']['sum']
        sum_s_idx = p['columns']['sum_of_squares']
        count_idx = p['columns']['count']
        row[sum_idx] = 100 * 51
        row[sum_s_idx] = 100 * (49 * 49)
        row[count_idx] = 100
        a = acc.stddev(alpha=0.05, properties=p, row=row)
        assert(a > 10.0 and a < 74.22634)
        a = acc.stddev(alpha=0.01, properties=p, row=row)
        assert(a > 10.0 and a < 81.44979)


class TestAccuracyDetect:
    def test_detect(self):
        assert(acc.properties[0]['statistic'] == 'mean')
        assert(acc.properties[1]['statistic'] == 'stddev')
        assert(acc.properties[2]['statistic'] == 'variance')
        assert(acc.properties[3]['statistic'] == 'sum')
        assert(acc.properties[4]['statistic'] == 'count')

class TestExecution:
    def test_no_accuracy(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            res = test_databases.to_tuples(reader.execute(query))
            assert(len(res) == 3)
            assert(all([len(r) == 5 for r in res]))
    def test_accuracy(self, test_databases):
        readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
        assert(len(readers) > 0)
        for reader in readers:
            if reader.engine == "spark":
                continue
            res = reader.execute_with_accuracy(query)
            assert(len(res) == 3)
            for row, accuracy in res:
                assert(len(row) == 5)
                assert(len(accuracy) == 2)
                acc99, acc95 = accuracy
                assert(all([a99 > a95 for a99, a95 in zip(acc99, acc95)]))
    def test_pandas_df_accuracy(self, test_databases):
        reader = test_databases.get_private_reader(database='PUMS_pid', engine="pandas", privacy=privacy)
        if reader is None:
            return # SKIP_PANDAS
        else:
            res = reader.execute_with_accuracy_df(query)
            df, accuracies = res
            acc99, acc95 = accuracies
            assert(len(df) == 2)
            assert(len(acc99) == 2)
            assert(len(acc95) == 2)
            assert(len(df.columns) == 5)
            assert(len(acc99.columns) == 5)
            assert(len(acc95.columns) == 5)
    def test_spark_accuracy(self, test_databases):
        priv = test_databases.get_private_reader(privacy=privacy, database="PUMS_pid", engine="spark")
        if priv is None:
            raise ValueError("No db available for spark")
            return # TEST_SPARK not set
        res = priv.execute_with_accuracy(query)
        row_count = 0
        for row, accuracies in res.collect():
            row_count += 1
            acc99, acc95 = accuracies
            assert(len(row) == 5)
            assert(len(acc99) == 5)
            assert(len(acc95) == 5)
        assert(row_count == 2) # PipelineRDD doesn't return column names

