# Implement different aggregation functions that can be passed through the verification tests
import random
import math
import numpy as np
import pandas as pd

import mlflow
import json
import sys

from burdock.query.sql.reader import CSVReader
from burdock.query.sql import MetadataLoader
from burdock.query.sql.private.query import PrivateQuery
from burdock.query.sql.reader.rowset import TypedRowset
from burdock.mechanisms.laplace import Laplace
from pandasql import sqldf

class Aggregation:
    def __init__(self, epsilon=1.0, t=1, repeat_count=10000):
        self.epsilon = epsilon
        self.t = t
        self.repeat_count = repeat_count

    # Taking df as a parameter it shall be passed both d1 and d2 that differ by 1 record
    def exact_count(self, df, colname):
        return np.zeros(self.repeat_count) + df[colname].count()

    def buggy_count(self, df, colname):
        return df[colname].count() + np.random.random_sample((self.repeat_count,))*10

    def dp_count(self, df, colname):
        delta = 1/(len(df) * math.sqrt(len(df)))
        sigmacnt = math.sqrt(self.t)*((math.sqrt(math.log(1/delta)) + math.sqrt(math.log((1/delta)) + self.epsilon)) / (math.sqrt(2)*self.epsilon))
        dp_noise = np.random.normal(0, sigmacnt, self.repeat_count)
        return df[colname].count() + dp_noise

    def dp_sum(self, df, colname):
        delta = 1/(len(df) * math.sqrt(len(df)))
        M = abs(max(df[colname]) - min(df[colname]))
        sigmasum = math.sqrt(self.t)*M*((math.sqrt(math.log(1/delta)) + math.sqrt(math.log((1/delta)) + self.epsilon)) / (math.sqrt(2)*self.epsilon))
        dp_noise = np.random.normal(0, sigmasum, self.repeat_count)
        return df[colname].sum() + dp_noise

    def dp_mean(self, df, colname):
        return np.divide(self.dp_sum(df, colname), self.dp_count(df, colname))

    def dp_var(self, df, colname):
        cnt = self.dp_count(df, colname)
        sum = self.dp_sum(df, colname)
        df[colname + "squared"] = df[colname] ** 2
        sumsq = self.dp_sum(df, colname + "squared")
        return np.subtract(np.divide(sumsq, cnt), np.power(np.divide(sum, cnt), 2))

    # Run the query using the private reader and input query
    # Get query response back
    def run_agg_query(self, df, metadata_path, query, confidence):
        metadata = MetadataLoader(metadata_path).read_schema()
        reader = CSVReader(metadata, df)
        private_reader = PrivateQuery(reader, metadata, self.epsilon)
        exact_values = private_reader._execute_exact(query)
        bounds_centered_zero = list(private_reader._apply_noise(*exact_values, confidence)[1].values())[1]
        actual_value = exact_values[-1].rows()[1:][0][1]
        bounds = np.array([bounds_centered_zero[0] + actual_value, bounds_centered_zero[1] + actual_value])
        noisy_values = np.array([private_reader._apply_noise(*exact_values)[0][1:][0][0] for i in range(self.repeat_count)])
        return noisy_values, bounds
