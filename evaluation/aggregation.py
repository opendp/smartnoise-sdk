# Implement different aggregation functions that can be passed through the verification tests
import random
import math
import numpy as np
import pandas as pd

import mlflow
import json
import sys
import os
import yarrow

from burdock.sql import PandasReader
from burdock.sql.private_query import PrivateReader
from burdock.reader.sql.rowset import TypedRowset
from burdock.mechanisms.laplace import Laplace
from burdock.mechanisms.gaussian import Gaussian
from pandasql import sqldf

class Aggregation:
    def __init__(self, epsilon=1.0, t=1, repeat_count=10000, mechanism="Laplace"):
        self.epsilon = epsilon
        self.t = t
        self.repeat_count = repeat_count
        self.mechanism = mechanism

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
        return np.array([mech.release([exact_count]).values[0] for i in range(self.repeat_count)])

    def dp_mechanism_sum(self, df, colname):
        exact_sum = df[colname].sum()
        M = float(abs(max(df[colname]) - min(df[colname])))
        mech = Laplace(self.epsilon, sensitivity = M)
        if(self.mechanism == "Gaussian"):
            mech = Gaussian(self.epsilon)
        return np.array([mech.release([exact_sum]).values[0] for i in range(self.repeat_count)])

    def dp_mechanism_mean(self, df, colname):
        return np.divide(self.dp_mechanism_sum(df, colname), self.dp_mechanism_count(df, colname))

    def dp_mechanism_var(self, df, colname):
        cnt = self.dp_mechanism_count(df, colname)
        sum = self.dp_mechanism_sum(df, colname)
        df[colname + "squared"] = df[colname] ** 2
        sumsq = self.dp_mechanism_sum(df, colname + "squared")
        return np.subtract(np.divide(sumsq, cnt), np.power(np.divide(sum, cnt), 2))

    # Apply noise to input aggregation function using Yarrow library
    def yarrow_dp_agg(self, f, dataset_path, args, kwargs):
        with yarrow.Analysis() as analysis:
            df = yarrow.Dataset('df', dataset_path)
            agg = f(df[args], **kwargs)
        noisy_values = []
        for x in range(self.repeat_count):
            analysis.release()
            noisy_values.append(analysis.release_proto.values[6].values['data'].f64.data[0])
        return np.array(noisy_values)
    
    # Apply noise to functions like covariance using Yarrow library that work on multiple columns
    def yarrow_dp_multi_agg(self, f, dataset_path, args, kwargs):
        with yarrow.Analysis() as analysis:
            df = yarrow.Dataset('df', dataset_path)
            agg = f(df[(args[0], args[1])], df[(args[2], args[3])], **kwargs)
        noisy_values = []
        for x in range(self.repeat_count):
            analysis.release()
            noisy_values.append(analysis.release_proto.values[10].values['data'].f64.data[0])
        return np.array(noisy_values)

    # Run the query using the private reader and input query
    # Get query response back
    def run_agg_query(self, df, metadata, query, confidence):
        reader = PandasReader(metadata, df)
        private_reader = PrivateReader(reader, metadata, self.epsilon)
        query_ast = private_reader.parse_query_string(query)
        subquery, query, syms, types, sens, srs_orig = private_reader._preprocess(query_ast)
        
        noisy_values = []
        for idx in range(self.repeat_count):
            srs = TypedRowset(srs_orig.rows(), types, sens)
            noisy_values.append(private_reader._postprocess(subquery, query, syms, types, sens, srs).rows()[1:][0][0])
        return np.array(noisy_values)
        
    # Run the query using the private reader and input query
    # Get query response back
    def run_agg_query_df(self, df, metadata, query, confidence, file_name = "d1"):
        reader = PandasReader(metadata, df)
        private_reader = PrivateReader(reader, metadata, self.epsilon)
        query_ast = private_reader.parse_query_string(query)
        subquery, query, syms, types, sens, srs_orig = private_reader._preprocess(query_ast)
        
        srs = TypedRowset(srs_orig.rows(), types, sens)
        sample_res = private_reader._postprocess(subquery, query, syms, types, sens, srs)
        headers = sample_res.colnames

        dim_cols = []
        num_cols = []

        for col in headers:
            if(sample_res.types[col] == "string"):
                dim_cols.append(col)
            else:
                num_cols.append(col)
        
        if(len(dim_cols) == 0):
            dim_cols.append("__dim__")
        
        res = []
        for idx in range(self.repeat_count):
            srs = TypedRowset(srs_orig.rows(), types, sens)
            singleres = private_reader._postprocess(subquery, query, syms, types, sens, srs).rows()[1:]
            for row in singleres:
                res.append(row)
        noisy_df = pd.DataFrame(res, columns=headers)
        
        if(dim_cols[0] == "__dim__"):
            noisy_df[dim_cols[0]] = ["key"]*len(noisy_df)

        return noisy_df, dim_cols, num_cols