# Implement different aggregation functions that can be passed through the verification tests
import random
import math
import numpy as np
import pandas as pd

import mlflow
import json
import sys
import os

from burdock.query.sql.reader import CSVReader
from burdock.query.sql.private.query import PrivateQuery
from burdock.query.sql.reader.rowset import TypedRowset
from burdock.mechanisms.laplace import Laplace
from burdock.mechanisms.gaussian import Gaussian
from pandasql import sqldf

class Aggregation:
    def __init__(self, epsilon=1.0, t=1, repeat_count=10000, mechanism="Laplace"):
        self.epsilon = epsilon
        self.t = t
        self.repeat_count = repeat_count
        self.mechanism = mechanism
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = r'../service/datasets/evaluation'

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

    def dp_mechanism_count(self, df, colname):
        exact_count = df[colname].count()
        mech = Laplace(self.epsilon)
        if(self.mechanism == "Gaussian"):
            mech = Gaussian(self.epsilon)
        return np.array([mech.release([exact_count])[0] for i in range(self.repeat_count)])

    def dp_mechanism_sum(self, df, colname):
        exact_sum = df[colname].sum()
        M = float(abs(max(df[colname]) - min(df[colname])))
        mech = Laplace(self.epsilon, sensitivity = M)
        if(self.mechanism == "Gaussian"):
            mech = Gaussian(self.epsilon)
        return np.array([mech.release([exact_sum])[0] for i in range(self.repeat_count)])

    def dp_mechanism_mean(self, df, colname):
        return np.divide(self.dp_mechanism_sum(df, colname), self.dp_mechanism_count(df, colname))

    def dp_mechanism_var(self, df, colname):
        cnt = self.dp_mechanism_count(df, colname)
        sum = self.dp_mechanism_sum(df, colname)
        df[colname + "squared"] = df[colname] ** 2
        sumsq = self.dp_mechanism_sum(df, colname + "squared")
        return np.subtract(np.divide(sumsq, cnt), np.power(np.divide(sum, cnt), 2))

    # Run the query using the private reader and input query
    # Get query response back
    def run_agg_query(self, df, metadata, query, confidence):
        reader = CSVReader(metadata, df)
        private_reader = PrivateQuery(reader, metadata, self.epsilon)
        query_ast = private_reader.parse_query_string(query)
        subquery, query, syms, types, sens, srs_orig = private_reader._preprocess(query_ast)
        
        noisy_values = []
        for idx in range(self.repeat_count):
            srs = TypedRowset(srs_orig.rows(), types, sens)
            noisy_values.append(private_reader._postprocess(subquery, query, syms, types, sens, srs).rows()[1:][0][0])
        return np.array(noisy_values)
        
    # Run the query using the private reader and input query
    # Get query response back
    def run_agg_query_file(self, df, metadata, query, confidence, file_name = "d1"):
        reader = CSVReader(metadata, df)
        private_reader = PrivateQuery(reader, metadata, self.epsilon)
        query_ast = private_reader.parse_query_string(query)
        subquery, query, syms, types, sens, srs_orig = private_reader._preprocess(query_ast)
        srs = TypedRowset(srs_orig.rows(), types, sens)
        headers = private_reader._postprocess(subquery, query, syms, types, sens, srs).rows()[0]

        file_path = os.path.join(self.file_dir, self.csv_path, file_name + ".csv")
        f = open(file_path, "w")
        f.write(','.join([str(n) for n in headers]) + "\n")

        for idx in range(self.repeat_count):
            srs = TypedRowset(srs_orig.rows(), types, sens)
            res = private_reader._postprocess(subquery, query, syms, types, sens, srs).rows()[1:]
            for row in res:
                f.write(','.join([str(n) for n in row]) + "\n")
        f.close()
        return
