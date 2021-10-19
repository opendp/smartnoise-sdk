import logging
test_logger = logging.getLogger("sql-test-logger")
from sneval.params._privacy_params import PrivacyParams
from sneval.params._eval_params import EvaluatorParams
from sneval.params._benchmark_params import BenchmarkParams
from sneval.params._dataset_params import DatasetParams
from sneval.report._report import Report
from sneval.privacyalgorithm._base import PrivacyAlgorithm
from sneval.evaluator._dp_evaluator import DPEvaluator
from sneval.benchmarking._dp_benchmark import DPBenchmarking
from sneval.metrics._metrics import Metrics
from snsql.sql import PandasReader
from snsql.metadata import *
from dp_singleton_query import DPSingletonQuery
import pytest
import pandas as pd
import numpy as np
import copy

class TestSql:
    def create_simulated_dataset(self, dataset_size, file_name):
        """
        Returns a simulated dataset of configurable size and following
        geometric distribution. Adds a couple of dimension columns for
        algorithm related to GROUP BY queries.
        """
        np.random.seed(1)
        userids = list(range(1, dataset_size+1))
        userids = ["A" + str(user) for user in userids]
        segment = ['A', 'B', 'C']
        role = ['R1', 'R2']
        roles = np.random.choice(role, size=dataset_size, p=[0.7, 0.3]).tolist()
        segments = np.random.choice(segment, size=dataset_size, p=[0.5, 0.3, 0.2]).tolist()
        usage = np.random.geometric(p=0.5, size=dataset_size).tolist()
        df = pd.DataFrame(list(zip(userids, segments, roles, usage)), columns=['UserId', 'Segment', 'Role', 'Usage'])

        # Storing the data as a CSV
        metadata = Table(file_name, file_name,  \
            [\
                String("UserId", dataset_size, True), \
                String("Segment", 3, False), \
                String("Role", 2, False), \
                Int("Usage", 0, 25)
            ], dataset_size)

        return df, metadata

    def generate_neighbors(self, df, metadata):
        """
        Generate dataframes that differ by a single record that is randomly chosen
        Returns the neighboring datasets and their corresponding metadata
        """
        d1 = df
        drop_idx = np.random.choice(df.index, 1, replace=False)
        d2 = df.drop(drop_idx)
        d1_table = metadata
        d2_table = copy.copy(d1_table)
        d1_table.schema, d2_table.schema = "dataset", "dataset"
        d1_table.name, d2_table.name = "dataset", "dataset"
        d2_table.rowcount = d1_table.rowcount - 1
        d1_metadata, d2_metadata = Metadata([d1_table], "csv"), Metadata([d2_table], "csv")

        return d1, d2, d1_metadata, d2_metadata

    def test_interface_count(self):
        logging.getLogger().setLevel(logging.DEBUG)
        # Initialize params and algorithm to benchmark
        pa = DPSingletonQuery()
        pp = PrivacyParams(epsilon=1.0)
        ev = EvaluatorParams(repeat_count=100)
        dd = DatasetParams(dataset_size=500)
        query = "SELECT COUNT(UserId) AS UserCount FROM dataset.dataset"

        # Preparing neighboring datasets
        df, metadata = self.create_simulated_dataset(dd.dataset_size, "dataset")
        d1_dataset, d2_dataset, d1_metadata, d2_metadata = self.generate_neighbors(df, metadata)
        d1 = PandasReader(d1_dataset, d1_metadata)
        d2 = PandasReader(d2_dataset, d2_metadata)

        # Call evaluate
        eval = DPEvaluator()
        key_metrics = eval.evaluate([d1_metadata, d1], [d2_metadata, d2], pa, query, pp, ev)
        # After evaluation, it should return True and distance metrics should be non-zero
        for key, metrics in key_metrics.items():
            assert(metrics.dp_res == True)
            test_logger.debug("Wasserstein Distance:" + str(metrics.wasserstein_distance))
            test_logger.debug("Jensen Shannon Divergence:" + str(metrics.jensen_shannon_divergence))
            test_logger.debug("KL Divergence:" + str(metrics.kl_divergence))
            test_logger.debug("MSE:" + str(metrics.mse))
            test_logger.debug("Standard Deviation:" + str(metrics.std))
            test_logger.debug("Mean Signed Deviation:" + str(metrics.msd))
            assert(metrics.wasserstein_distance > 0.0)
            assert(metrics.jensen_shannon_divergence > 0.0)
            assert(metrics.kl_divergence != 0.0)
            assert(metrics.mse > 0.0)
            assert(metrics.std != 0.0)
            assert(metrics.msd != 0.0)
