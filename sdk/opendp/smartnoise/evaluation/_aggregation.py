import random
import math
import numpy as np
import pandas as pd

import json
import sys
import os

from opendp.smartnoise.sql import PandasReader, PrivateReader
from opendp.smartnoise.sql._mechanisms.gaussian import Gaussian
from pandasql import sqldf

class Aggregation:
    """
    Implement different aggregation functions that can be passed through
    the verification tests
    """
    def __init__(self, epsilon=1.0, t=1, repeat_count=10000):
        self.epsilon = epsilon
        self.t = t
        self.repeat_count = repeat_count

    def exact_count(self, df, colname):
        """
        Exact count taking df as a parameter it shall be passed both d1 and d2
        that differ by 1 record
        """
        return np.zeros(self.repeat_count) + df[colname].count()

    def buggy_count(self, df, colname):
        """
        Example of non-DP noise repeatedly applied while
        counting records in a dataset
        """
        return df[colname].count() + np.random.random_sample((self.repeat_count,))*10

    def dp_count(self, df, colname):
        """
        Returns repeatedly applied differentially private noisy responses while
        counting rows of a column in a dataset
        """
        delta = 1/(len(df) * math.sqrt(len(df)))
        sigmacnt = math.sqrt(self.t)*((math.sqrt(math.log(1/delta)) + math.sqrt(math.log((1/delta)) + self.epsilon)) / (math.sqrt(2)*self.epsilon))
        dp_noise = np.random.normal(0, sigmacnt, self.repeat_count)
        return df[colname].count() + dp_noise

    def dp_sum(self, df, colname):
        """
        Returns repeatedly applied differentially private noisy response to
        summing rows of a numerical column in a dataset
        """
        delta = 1/(len(df) * math.sqrt(len(df)))
        M = abs(max(df[colname]) - min(df[colname]))
        sigmasum = math.sqrt(self.t)*M*((math.sqrt(math.log(1/delta)) + math.sqrt(math.log((1/delta)) + self.epsilon)) / (math.sqrt(2)*self.epsilon))
        dp_noise = np.random.normal(0, sigmasum, self.repeat_count)
        return df[colname].sum() + dp_noise

    def dp_mean(self, df, colname):
        """
        Returns repeatedly applied differentially private noisy response to
        averaging rows of a numerical column in a dataset
        """
        return np.divide(self.dp_sum(df, colname), self.dp_count(df, colname))

    def dp_var(self, df, colname):
        """
        Returns repeatedly applied differentially private noisy response to
        calculating variance of a numerical column in a dataset
        """
        cnt = self.dp_count(df, colname)
        sum = self.dp_sum(df, colname)
        df[colname + "squared"] = df[colname] ** 2
        sumsq = self.dp_sum(df, colname + "squared")
        return np.subtract(np.divide(sumsq, cnt), np.power(np.divide(sum, cnt), 2))

    def dp_mechanism_count(self, df, colname):
        """
        Returns repeatedly applied noise adding mechanism to count query
        """
        exact_count = df[colname].count()
        mech = Gaussian(self.epsilon)
        return np.array([mech.release([exact_count]).values[0] for i in range(self.repeat_count)])

    def dp_mechanism_sum(self, df, colname):
        """
        Returns repeatedly applied noise adding mechanisms to sum query.
        Sensitivity is set as absolute difference between max and min values
        within the column
        """
        exact_sum = df[colname].sum()
        M = float(abs(max(df[colname]) - min(df[colname])))
        mech = Gaussian(self.epsilon, 10E-5, M)
        return np.array([mech.release([exact_sum]).values[0] for i in range(self.repeat_count)])

    def dp_mechanism_mean(self, df, colname):
        """
        Returns repeatedly applied noise adding mechanisms to
        AVG query by dividing noisy response to SUM by noisy response to COUNT query
        """
        return np.divide(self.dp_mechanism_sum(df, colname), self.dp_mechanism_count(df, colname))

    def dp_mechanism_var(self, df, colname):
        """
        Returns repeatedly applied noise adding mechanisms to
        VAR query by internally using results of DP-SUM and DP-COUNT queries
        """
        cnt = self.dp_mechanism_count(df, colname)
        sum = self.dp_mechanism_sum(df, colname)
        df[colname + "squared"] = df[colname] ** 2
        sumsq = self.dp_mechanism_sum(df, colname + "squared")
        return np.subtract(np.divide(sumsq, cnt), np.power(np.divide(sum, cnt), 2))

    def run_agg_query(self, df, metadata, query, confidence, get_exact=True):
        """
        Run the query using the private reader and input query
        Get query response back
        """
        reader = PandasReader(df, metadata)
        actual = 0.0
        # VAR not supported in Pandas Reader. So not needed to fetch actual on every aggregation
        if(get_exact):
            actual = reader.execute(query)[1:][0][0]
        private_reader = PrivateReader(reader, metadata, self.epsilon)
        query_ast = private_reader.parse_query_string(query)

        noisy_values = []
        low_bounds = []
        high_bounds = []
        for idx in range(self.repeat_count):
            res = private_reader._execute_ast(query_ast, True)
            # Disabled because confidence interval not available in report
            #interval = res.report[res.colnames[0]].intervals[confidence]
            #low_bounds.append(interval[0].low)
            #high_bounds.append(interval[0].high)
            noisy_values.append(res[1:][0][0])
        return np.array(noisy_values), actual, low_bounds, high_bounds

    def run_agg_query_df(self, df, metadata, query, confidence, file_name = "d1"):
        """
        Run the query using the private reader and input query
        Get query response back for multiple dimensions and aggregations
        """
        # Getting exact result
        reader = PandasReader(df, metadata)
        exact_res = reader.execute(query)[1:]

        private_reader = PrivateReader(reader, metadata, self.epsilon)
        query_ast = private_reader.parse_query_string(query)

        # Distinguishing dimension and measure columns

        sample_res = private_reader._execute_ast(query_ast, True)
        headers = sample_res[0]

        dim_cols = []
        num_cols = []

        out_syms = query_ast.all_symbols()
        out_types = [s[1].type() for s in out_syms]
        out_col_names = [s[0] for s in out_syms]

        for col, ctype in zip(out_col_names, out_types):
            if(ctype == "string"):
                dim_cols.append(col)
            else:
                num_cols.append(col)

        # Repeated query and store results
        res = []
        for idx in range(self.repeat_count):
            dim_rows = []
            num_rows = []
            singleres = private_reader._execute_ast_df(query_ast, True)
            #values = singleres[col]
            for col in dim_cols:
                dim_rows.append(singleres[col].tolist())
            for col in num_cols:
                values = singleres[col].tolist()
                num_rows.append(list(zip(values)))

            res.extend(list(zip(*dim_rows, *num_rows)))

        exact_df = pd.DataFrame(exact_res, columns=headers)
        noisy_df = pd.DataFrame(res, columns=headers)

        # Add a dummy dimension column for cases where no dimensions available for merging D1 and D2
        if(len(dim_cols) == 0):
            dim_cols.append("__dim__")

        if(dim_cols[0] == "__dim__"):
            exact_df[dim_cols[0]] = ["key"]*len(exact_df)
            noisy_df[dim_cols[0]] = ["key"]*len(noisy_df)

        return noisy_df, exact_df, dim_cols, num_cols
