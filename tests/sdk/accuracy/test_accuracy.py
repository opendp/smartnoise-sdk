import os
import subprocess
from opendp.smartnoise.sql._mechanisms.accuracy import Accuracy
from pyspark.sql import SparkSession
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
query = 'SELECT AVG(age), STD(age), VAR(age), SUM(age), COUNT(age) FROM PUMS.PUMS GROUP BY sex'
q = QueryParser(meta).query(query)

privacy = Privacy(alphas=[0.01, 0.05], delta=1/(math.sqrt(100) * 100))
priv = PrivateReader.from_connection(pums, privacy=privacy, metadata=meta)
subquery, root = priv._rewrite(query)

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
        p = p[0]
        row = [0, 0, 0, 0, 0]
        sum_idx = p['columns']['sum']
        count_idx = p['columns']['count']
        row[sum_idx] = 100 * 50
        row[count_idx] = 100
        a = acc.mean(alpha=0.05, properties=p, row=row)
        assert(np.isclose(a, 25.3707))
        a = acc.mean(alpha=0.01, properties=p, row=row)
        assert(np.isclose(a, 30.1797))
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
        assert(np.isclose(a, 5211.43496))
        a = acc.variance(alpha=0.01, properties=p, row=row)
        assert(np.isclose(a, 6358.240726))
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
        assert(np.isclose(a, 72.1902692))
        a = acc.stddev(alpha=0.01, properties=p, row=row)
        assert(np.isclose(a, 79.7385774))


class TestAccuracyDetect:
    def test_detect(self):
        assert(acc.properties[0]['statistic'] == 'mean')
        assert(acc.properties[1]['statistic'] == 'stddev')
        assert(acc.properties[2]['statistic'] == 'variance')
        assert(acc.properties[3]['statistic'] == 'sum')
        assert(acc.properties[4]['statistic'] == 'count')

class TestExecution:
    def test_no_accuracy(self):
        res = priv.execute(query)
        assert(len(res) == 3)
        assert(all([len(r) == 5 for r in res]))
    def test_accuracy(self):
        res = priv.execute_with_accuracy(query)
        assert(len(res) == 3)
        for row, accuracy in res:
            assert(len(row) == 5)
            assert(len(accuracy) == 2)
            acc99, acc95 = accuracy
            assert(all([a99 > a95 for a99, a95 in zip(acc99, acc95)]))
    def test_pandas_df_accuracy(self):
        res = priv.execute_with_accuracy_df(query)
        df, accuracies = res
        acc99, acc95 = accuracies
        assert(len(df) == 2)
        assert(len(acc99) == 2)
        assert(len(acc95) == 2)
        assert(len(df.columns) == 5)
        assert(len(acc99.columns) == 5)
        assert(len(acc95.columns) == 5)
    def test_spark_accuracy(self):
        spark = SparkSession.builder.getOrCreate()
        pums = spark.read.load(csv_path, format="csv", sep=",",inferSchema="true", header="true")
        query_modified = query.replace("PUMS.PUMS", "PUMS")
        pums.createOrReplaceTempView("PUMS")
        priv = PrivateReader.from_connection(spark, metadata=meta, privacy=privacy)
        priv.reader.compare.search_path = ["PUMS"]
        res = priv.execute_with_accuracy(query_modified)
        row_count = 0
        for row, accuracies in res.collect():
            row_count += 1
            acc99, acc95 = accuracies
            assert(len(row) == 5)
            assert(len(acc99) == 5)
            assert(len(acc95) == 5)
        assert(row_count == 2) # PipelineRDD doesn't return column names


