# This file contains a list of tests that can be passed actual aggregates or result aggregates from a DP implementation
# It tries to use a sample dataset S and splits it randomly into two neighboring datasets D1 and D2
# Using these neighboring datasets, it applies the aggregate query repeatedly
# It tests the DP condition to let the DP implementer know whether repeated aggregate query results are not enough to re-identify D1 or D2 which differ by single individual
# i.e. passing (epsilon, delta) - DP condition
# If the definition is not passed, there is a bug or it is a by-design bug in case of passing actual aggregates

import sys
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import opendp.whitenoise.evaluation.aggregation as agg
import opendp.whitenoise.evaluation.exploration as exp
import copy

# import yarrow
from opendp.whitenoise.metadata.collection import *
from scipy import stats

class DPVerification:
    # Set the epsilon parameter of differential privacy
    def __init__(self, epsilon=1.0, dataset_size=10000, csv_path="."):
        self.epsilon = epsilon
        self.dataset_size = dataset_size
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = csv_path
        self.df, self.dataset_path, self.file_name, self.metadata = self.create_simulated_dataset()
        print("Loaded " + str(len(self.df)) + " records")
        self.N = len(self.df)
        self.delta = 1/(self.N * math.sqrt(self.N))

    def create_simulated_dataset(self, file_name = "simulation"):
        np.random.seed(1)
        userids = list(range(1, self.dataset_size+1))
        userids = ["A" + str(user) for user in userids]
        segment = ['A', 'B', 'C']
        role = ['R1', 'R2']
        roles = np.random.choice(role, size=self.dataset_size, p=[0.7, 0.3]).tolist()
        segments = np.random.choice(segment, size=self.dataset_size, p=[0.5, 0.3, 0.2]).tolist()
        usage = np.random.geometric(p=0.5, size=self.dataset_size).tolist()
        df = pd.DataFrame(list(zip(userids, segments, roles, usage)), columns=['UserId', 'Segment', 'Role', 'Usage'])

        # Storing the data as a CSV
        file_path = os.path.join(self.file_dir, self.csv_path, file_name + ".csv")
        df.to_csv(file_path, sep=',', encoding='utf-8', index=False)
        metadata = Table(file_name, file_name, self.dataset_size, \
            [\
                String("UserId", self.dataset_size, True), \
                String("Segment", 3, False), \
                String("Role", 2, False), \
                Int("Usage", 0, 25)
            ])

        return df, file_path, file_name, metadata

    # Generate dataframes that differ by a single record that is randomly chosen
    def generate_neighbors(self, load_csv = False):
        if(load_csv):
            self.df = pd.read_csv(self.dataset_path)

        if(self.N == 0):
            print("No records in dataframe to run the test")
            return None, None

        d1 = self.df
        drop_idx = np.random.choice(self.df.index, 1, replace=False)
        d2 = self.df.drop(drop_idx)
        print("Length of D1: ", len(d1), " Length of D2: ", len(d2))

        if(load_csv):
            # Storing the data as a CSV for applying queries via Burdock querying system
            d1_file_path = os.path.join(self.file_dir, self.csv_path , "d1.csv")
            d2_file_path = os.path.join(self.file_dir, self.csv_path , "d2.csv")

            d1.to_csv(d1_file_path, sep=',', encoding='utf-8', index=False)
            d2.to_csv(d2_file_path, sep=',', encoding='utf-8', index=False)

        d1_table = self.metadata
        d2_table = copy.copy(d1_table)
        d1_table.schema, d2_table.schema = "d1", "d2"
        d1_table.name, d2_table.name = "d1", "d2"
        d2_table.rowcount = d1_table.rowcount - 1
        d1_metadata, d2_metadata = CollectionMetadata([d1_table], "csv"), CollectionMetadata([d2_table], "csv")

        return d1, d2, d1_metadata, d2_metadata

    # If there is an aggregation function that we need to test, we need to apply it on neighboring datasets
    # This function applies the aggregation repeatedly to log results in two vectors that are then used for generating histogram
    # The histogram is then passed through the DP test
    def apply_aggregation_neighbors(self, f, args1, args2):
        fD1 = f(*args1)
        fD2 = f(*args2)

        print("Mean fD1: ", np.mean(fD1), " Stdev fD1: ", np.std(fD1), " Mean fD2: ", np.mean(fD2), " Stdev fD2: ", np.std(fD2))
        return fD1, fD2

    # Generate histograms given the vectors of repeated aggregation results applied on neighboring datasets
    def generate_histogram_neighbors(self, fD1, fD2, numbins=0, binsize="auto", exact=False):
        d1 = fD1
        d2 = fD2
        d = np.concatenate((d1, d2), axis=None)
        n = d.size
        binlist = []
        minval = min(min(d1), min(d2))
        maxval = max(max(d1), max(d2))
        if(exact):
            binlist = np.linspace(minval, maxval, 2)
        elif(numbins > 0):
            binlist = np.linspace(minval, maxval, numbins)
        elif(binsize == "auto"):
            iqr = np.subtract(*np.percentile(d, [75, 25]))
            numerator = 2 * iqr if iqr > 0 else maxval - minval
            denominator = n ** (1. / 3)
            binwidth = numerator / denominator # Freedman–Diaconis' choice
            numbins = int(math.ceil((maxval - minval) / binwidth)) if maxval > minval else 20
            binlist = np.linspace(minval, maxval, numbins)
        else:
            # Choose bin size of unity
            binlist = np.arange(np.floor(minval),np.ceil(maxval))

        # Calculating histograms of fD1 and fD2
        d1hist, bin_edges = np.histogram(d1, bins = binlist, density = False)
        d2hist, bin_edges = np.histogram(d2, bins = binlist, density = False)

        return d1hist, d2hist, bin_edges

    # Plot histograms given the vectors of repeated aggregation results applied on neighboring datasets
    def plot_histogram_neighbors(self, fD1, fD2, d1histupperbound, d2histupperbound, d1hist, d2hist, d1lower, d2lower, binlist, bound=True, exact=False):
        plt.figure(figsize=(15,5))
        if(exact):
            ax = plt.subplot(1, 1, 1)
            ax.ticklabel_format(useOffset=False)
            plt.xlabel('Bin')
            plt.ylabel('Probability')
            plt.hist(fD1, width=0.2, alpha=0.5, ec="k", align = "right", bins = 1)
            plt.hist(fD2, width=0.2, alpha=0.5, ec="k", align = "right", bins = 1)
            ax.legend(['D1', 'D2'], loc="upper right")
            return

        ax = plt.subplot(1, 2, 1)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel('Bin')
        plt.ylabel('Probability')
        if(bound):
            plt.bar(binlist[:-1], d2histupperbound, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d1lower, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(['D1', 'D2'], loc="upper right")
        else:
            plt.bar(binlist[:-1], d1hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d2hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(['D1', 'D2'], loc="upper right")

        ax = plt.subplot(1, 2, 2)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel('Bin')
        plt.ylabel('Probability')
        if(bound):
            plt.bar(binlist[:-1], d1histupperbound, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d2lower, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(['D2', 'D1'], loc="upper right")
        else:
            plt.bar(binlist[:-1], d2hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d1hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(['D2', 'D1'], loc="upper right")
        plt.show()

    # Check if histogram of fD1 values multiplied by e^epsilon and summed by delta is bounding fD2 and vice versa
    # Use the histogram results and create bounded histograms to compare in DP test
    def get_bounded_histogram(self, d1hist, d2hist, binlist, d1size, d2size, exact, alpha=0.05):
        d1_error_interval = 0.0
        d2_error_interval = 0.0
        # Lower and Upper bound
        if(not exact):
            num_buckets = binlist.size - 1
            critical_value = stats.norm.ppf(1-(alpha / 2 / num_buckets), loc=0.0, scale=1.0)
            d1_error_interval = critical_value * math.sqrt(num_buckets / d1size) / 2
            d2_error_interval = critical_value * math.sqrt(num_buckets / d2size) / 2

        num_buckets = binlist.size - 1
        px = np.divide(d1hist, d1size)
        py = np.divide(d2hist, d2size)

        d1histbound = px * math.exp(self.epsilon) + self.delta
        d2histbound = py * math.exp(self.epsilon) + self.delta

        d1upper = np.power(np.sqrt(px * num_buckets) + d1_error_interval, 2) / num_buckets
        d2upper = np.power(np.sqrt(py * num_buckets) + d2_error_interval, 2) / num_buckets
        d1lower = np.power(np.sqrt(px * num_buckets) - d1_error_interval, 2) / num_buckets
        d2lower = np.power(np.sqrt(py * num_buckets) - d2_error_interval, 2) / num_buckets

        np.maximum(d1lower, 0.0, d1lower)
        np.maximum(d2lower, 0.0, d1lower)

        d1histupperbound = d1upper * math.exp(self.epsilon) + self.delta
        d2histupperbound = d2upper * math.exp(self.epsilon) + self.delta

        return px, py, d1histupperbound, d2histupperbound, d1histbound, d2histbound, d1lower, d2lower

    # Differentially Private Predicate Test
    def dp_test(self, d1hist, d2hist, binlist, d1size, d2size, debug=False, exact=False):
        px, py, d1histupperbound, d2histupperbound, d1histbound, d2histbound, d1lower, d2lower = \
            self.get_bounded_histogram(d1hist, d2hist, binlist, d1size, d2size, exact)
        if(debug):
            print("Parameters")
            print("epsilon: ", self.epsilon, " delta: ", self.delta)
            print("Bins\n", binlist)
            print("Original D1 Histogram\n", d1hist)
            print("Probability of D1 Histogram\n", px)
            print("D1 Lower\n", d1lower)
            print("D1 Upper\n", d1histupperbound)
            print("D1 Histogram to bound D2\n", d1histbound)
            print("Original D2 Histogram\n", d2hist)
            print("Probability of D2 Histogram\n", py)
            print("D2 Lower\n", d2lower)
            print("D2 Upper\n", d2histupperbound)
            print("D2 Histogram to bound D1\n", d2histbound)
            print("Comparison - D2 bound to D1\n", np.greater(d1hist, np.zeros(d1hist.size)), np.logical_and(np.greater(d1hist, np.zeros(d1hist.size)), np.greater(d1lower, d2histupperbound)))
            print("Comparison - D1 bound to D2\n", np.greater(d2hist, np.zeros(d2hist.size)), np.logical_and(np.greater(d2hist, np.zeros(d2hist.size)), np.greater(d2lower, d1histupperbound)))

        # Check if any of the bounds across the bins violate the relaxed DP condition
        bound_exceeded = np.any(np.logical_and(np.greater(d1hist, np.zeros(d1hist.size)), np.greater(d1lower, d2histupperbound))) or \
        np.any(np.logical_and(np.greater(d2hist, np.zeros(d2hist.size)), np.greater(d2lower, d1histupperbound)))
        return not bound_exceeded, d1histupperbound, d2histupperbound, d1lower, d2lower

    # K-S Two sample test between the repeated query results on neighboring datasets
    def ks_test(self, fD1, fD2):
        return stats.ks_2samp(fD1, fD2)

    # Anderson Darling Test
    def anderson_ksamp(self, fD1, fD2):
        return stats.anderson_ksamp([fD1, fD2])

    # Kullback-Leibler divergence D(P || Q) for discrete distributions
    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    # Wasserstein Distance
    def wasserstein_distance(self, d1hist, d2hist):
        return stats.wasserstein_distance(d1hist, d2hist)

    # Verification of SQL aggregation mechanisms
    def aggtest(self, f, colname, numbins=0, binsize="auto", debug=False, plot=True, bound=True, exact=False):
        d1, d2, d1_metadata, d2_metadata = self.generate_neighbors()

        fD1, fD2 = self.apply_aggregation_neighbors(f, (d1, colname), (d2, colname))
        d1size, d2size = fD1.size, fD2.size

        ks_res = self.ks_test(fD1, fD2)
        print("\nKS 2-sample Test Result: ", ks_res, "\n")

        #andderson_res = self.anderson_ksamp(fD1, fD2)
        #print("Anderson 2-sample Test Result: ", andderson_res, "\n")

        d1hist, d2hist, bin_edges = \
            self.generate_histogram_neighbors(fD1, fD2, numbins, binsize, exact=exact)

        ws_res = 0.0
        #kl_res = 0.0
        dp_res, d1histupperbound, d2histupperbound, d1lower, d2lower = self.dp_test(d1hist, d2hist, bin_edges, d1size, d2size, debug, exact=exact)
        if(exact):
            dp_res = False
            print("Wasserstein Distance: ", ws_res, "\n")
            #print("KL Divergence Distance: ", kl_res, "\n")
        else:
            ws_res = self.wasserstein_distance(d1hist, d2hist)
            print("Wasserstein Distance: ", ws_res, "\n")
            #kl_res = self.kl_divergence(d1histupperbound, d2lower)
            #print("KL-Divergence: ", kl_res, "\n")
        print("DP Predicate Test:", dp_res, "\n")

        if(plot):
            self.plot_histogram_neighbors(fD1, fD2, d1histupperbound, d2histupperbound, d1hist, d2hist, d1lower, d2lower, bin_edges, bound, exact)
        return dp_res, ks_res, ws_res

    # # Verification of aggregation mechanisms implemented in Yarrow
    # # Creating a new function to take in non-keyworded args and keyworded kwargs
    # # This makes it generic to take in any Yarrow aggregate function with any set of parameters
    # # DP-SQL queries in Burdock use other aggregation functions in Aggregation class
    # def yarrow_test(self, dataset_path, f, *args, numbins=0, binsize="auto", debug=False, plot=True, bound=True, exact=False, repeat_count=10000, **kwargs):
    #     ag = agg.Aggregation(t=1, repeat_count=repeat_count)
    #     self.dataset_path = dataset_path
    #     d1, d2 = self.generate_neighbors(load_csv=True)

    #     d1_file_path = os.path.join(self.file_dir, self.csv_path , "d1.csv")
    #     d2_file_path = os.path.join(self.file_dir, self.csv_path , "d2.csv")

    #     if(len(args) == 4):
    #         fD1 = ag.yarrow_dp_multi_agg(f, d1_file_path, args, kwargs)
    #         fD2 = ag.yarrow_dp_multi_agg(f, d2_file_path, args, kwargs)
    #     else:
    #         fD1 = ag.yarrow_dp_agg(f, d1_file_path, args, kwargs)
    #         fD2 = ag.yarrow_dp_agg(f, d2_file_path, args, kwargs)

    #     d1size, d2size = fD1.size, fD2.size
    #     d1hist, d2hist, bin_edges = \
    #         self.generate_histogram_neighbors(fD1, fD2, numbins, binsize, exact=exact)
    #     dp_res, d1histupperbound, d2histupperbound, d1lower, d2lower = self.dp_test(d1hist, d2hist, bin_edges, d1size, d2size, debug)
    #     print("DP Predicate Test:", dp_res, "\n")

    #     if(plot):
    #         self.plot_histogram_neighbors(fD1, fD2, d1histupperbound, d2histupperbound, d1hist, d2hist, d1lower, d2lower, bin_edges, bound)
    #     return dp_res

    def accuracy_test(self, actual, low, high, confidence=0.95):
        # Actual mean of aggregation function f on D1 is equal to sample mean
        n = len(low)
        actual = [actual] * n
        error_interval = 0.05*confidence
        relaxed_low = confidence - error_interval
        relaxed_high = 1 - (confidence + error_interval)
        within_bounds = np.sum(np.logical_and(np.greater_equal(actual, low), np.greater_equal(high, actual)))
        outside_bounds = n - within_bounds
        print("Count of times noisy result within bounds:", within_bounds, "/", n)
        print("Count of times noisy result outside bounds:", outside_bounds, "/", n)
        acc_res = (within_bounds / n >= relaxed_low)
        utility_res = (outside_bounds / n >= relaxed_high)
        return acc_res, utility_res, float('%.2f'%((within_bounds / n) * 100))

    # Calculates mean signed deviation from noisy results sample as a ratio of actual value
    def bias_test(self, actual, fD, sig_level = 0.05):
        # Mean signed deviation
        n = len(fD)
        actual = [actual] * n
        diff = fD - actual
        msd = (np.sum(diff) / n) / actual[0]
        print("Mean signed deviation ratio to actual: ", msd)
        # Checking if mean of (difference of noisy response to actual) is zero i.e. unbiased result
        tset, pval = stats.ttest_1samp(diff, 0.0)
        return (pval < sig_level), msd

    # Applying queries repeatedly against SQL-92 implementation of Differential Privacy by Burdock
    def dp_query_test(self, d1_query, d2_query, debug=False, plot=True, bound=True, exact=False, repeat_count=10000, confidence=0.95, get_exact=True):
        ag = agg.Aggregation(t=1, repeat_count=repeat_count)
        d1, d2, d1_metadata, d2_metadata = self.generate_neighbors(load_csv=True)

        fD1, fD1_actual, fD1_low, fD1_high = ag.run_agg_query(d1, d1_metadata, d1_query, confidence, get_exact)
        fD2, fD2_actual, fD2_low, fD2_high = ag.run_agg_query(d2, d2_metadata, d2_query, confidence, get_exact)
        d1hist, d2hist, bin_edges = self.generate_histogram_neighbors(fD1, fD2, binsize="auto")
        d1size, d2size = fD1.size, fD2.size
        dp_res, d1histupperbound, d2histupperbound, d1lower, d2lower = self.dp_test(d1hist, d2hist, bin_edges, d1size, d2size, debug)
        acc_res, utility_res, within_bounds = self.accuracy_test(fD1_actual, fD1_low, fD1_high, confidence)
        bias_res, msd = self.bias_test(fD1_actual, fD1)
        if(plot):
            self.plot_histogram_neighbors(fD1, fD2, d1histupperbound, d2histupperbound, d1hist, d2hist, d1lower, d2lower, bin_edges, bound, exact)
        return dp_res, acc_res, utility_res, bias_res

    # Allows DP Predicate test on both singleton and GROUP BY queries
    def dp_groupby_query_test(self, d1_query, d2_query, debug=False, plot=True, bound=True, exact=False, repeat_count=10000, confidence=0.95):
        ag = agg.Aggregation(t=1, repeat_count=repeat_count)
        d1, d2, d1_metadata, d2_metadata = self.generate_neighbors(load_csv=True)

        d1_res, d1_exact, dim_cols, num_cols = ag.run_agg_query_df(d1, d1_metadata, d1_query, confidence, file_name = "d1")
        d2_res, d2_exact, dim_cols, num_cols = ag.run_agg_query_df(d2, d2_metadata, d2_query, confidence, file_name = "d2")

        res_list = []
        for col in num_cols:
            d1_gp = d1_res.groupby(dim_cols)[col].apply(list).reset_index(name=col)
            d2_gp = d2_res.groupby(dim_cols)[col].apply(list).reset_index(name=col)
            exact_gp = d1_exact.groupby(dim_cols)[col].apply(list).reset_index(name=col)
            # Full outer join after flattening the results above to one row per dimension key
            # We cannot be sure if every dimension key has a response in every repeated query run because of tau thresholding
            # That's why we do a full outer join and flatten whatever vector of results we get for the numerical column across repeat runs
            # This is what we use for generating the histogram of results for that dimension key
            d1_d2 = d1_gp.merge(d2_gp, on=dim_cols, how='outer')
            d1_d2 = d1_d2.merge(exact_gp, on=dim_cols, how='left')
            n_cols = len(d1_d2.columns)
            for index, row in d1_d2.iterrows():
                print(d1_d2.iloc[index, :n_cols - 3])
                print("Column: ", col)
                # fD1 and fD2 will have the results of the K repeated query results that can be passed through histogram test
                # These results are for that particular numerical column and the specific dimension key of d1_d2
                fD1 = np.array([val[0] for val in d1_d2.iloc[index, n_cols - 3]])
                fD2 = np.array([val[0] for val in d1_d2.iloc[index, n_cols - 2]])
                exact_val = d1_d2.iloc[index, n_cols - 1][0]
                d1hist, d2hist, bin_edges = self.generate_histogram_neighbors(fD1, fD2, binsize="auto")
                d1size, d2size = fD1.size, fD2.size
                dp_res, d1histupperbound, d2histupperbound, d1lower, d2lower = self.dp_test(d1hist, d2hist, bin_edges, d1size, d2size, debug)
                print("DP Predicate Test Result: ", dp_res)

                # Accuracy Test
                low = np.array([val[1] for val in d1_d2.iloc[index, n_cols - 2]])
                high = np.array([val[2] for val in d1_d2.iloc[index, n_cols - 2]])
                acc_res, utility_res, within_bounds = self.accuracy_test(exact_val, low, high, confidence)
                bias_res, msd = self.bias_test(exact_val, fD1)
                res_list.append([dp_res, acc_res, utility_res, within_bounds, bias_res, msd])
                if(plot):
                    self.plot_histogram_neighbors(fD1, fD2, d1histupperbound, d2histupperbound, d1hist, d2hist, d1lower, d2lower, bin_edges, bound, exact)

        for res in res_list:
            print(res)

        res_list = res_list.values() if hasattr(res_list, "values") else res_list  # TODO why is this needed?
        dp_res = np.all(np.array([res[0] for res in res_list]))
        acc_res = np.all(np.array([res[1] for res in res_list]))
        utility_res = np.all(np.array([res[2] for res in res_list]))
        bias_res = np.all(np.array([res[4] for res in res_list]))
        return dp_res, acc_res, utility_res, bias_res

    # Use the powerset based neighboring datasets to scan through all edges of database search graph
    def dp_powerset_test(self, query_str, debug=False, plot=True, bound=True, exact=False, repeat_count=10000, confidence=0.95, test_cases=5):
        ag = agg.Aggregation(t=1, repeat_count=repeat_count)
        ex = exp.Exploration()
        res_list = {}
        halton_samples = ex.generate_halton_samples(bounds = ex.corners, dims = ex.N, n_sample=test_cases)
        # Iterate through each sample generated by halton sequence
        for sample in halton_samples:
            df, metadata = ex.create_small_dataset(sample)
            ex.generate_powerset(df)
            print("Test case: ", list(sample))
            for filename in ex.visited:
                print("Testing: ", filename)
                d1_query = query_str + "d1_" + filename + "." + "d1_" + filename
                d2_query = query_str + "d2_" + filename + "." + "d2_" + filename
                [d1, d2, d1_metadata, d2_metadata] = ex.neighbor_pair[filename]
                fD1, fD1_actual, fD1_low, fD1_high = ag.run_agg_query(d1, d1_metadata, d1_query, confidence)
                fD2, fD2_actual, fD2_low, fD2_high = ag.run_agg_query(d2, d2_metadata, d2_query, confidence)

                acc_res, utility_res, within_bounds = self.accuracy_test(fD1_actual, fD1_low, fD1_high, confidence)
                bias_res, msd = self.bias_test(fD1_actual, fD1)
                d1hist, d2hist, bin_edges = self.generate_histogram_neighbors(fD1, fD2, binsize="auto")
                d1size, d2size = fD1.size, fD2.size
                dp_res, d1histupperbound, d2histupperbound, d1lower, d2lower = self.dp_test(d1hist, d2hist, bin_edges, d1size, d2size, debug)
                print("DP Predicate Test Result: ", dp_res)
                if(plot):
                    self.plot_histogram_neighbors(fD1, fD2, d1histupperbound, d2histupperbound, d1hist, d2hist, d1lower, d2lower, bin_edges, bound, exact)
                key = "[" + ','.join(str(e) for e in list(sample)) + "] - " + filename
                res_list[key] = [dp_res, acc_res, utility_res, within_bounds, bias_res, msd]

        print("Halton sequence based Powerset Test Result")
        for data, res in res_list.items():
            print(data, "-", res)

        dp_res = np.all(np.array([res[0] for res in res_list]))
        acc_res = np.all(np.array([res[1] for res in res_list]))
        utility_res = np.all(np.array([res[2] for res in res_list]))
        bias_res = np.all(np.array([res[4] for res in res_list]))
        return dp_res, acc_res, utility_res, bias_res

    # Main method listing all the DP verification steps
    def main(self):
        # COUNT Example
        d1_query = "SELECT COUNT(UserId) AS UserCount FROM d1.d1"
        d2_query = "SELECT COUNT(UserId) AS UserCount FROM d2.d2"
        dp_res, acc_res, utility_res, bias_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=1000)

        d1_query = "SELECT SUM(Usage) AS Usage FROM d1.d1"
        d2_query = "SELECT SUM(Usage) AS Usage FROM d2.d2"
        dp_res, acc_res, utility_res, bias_res = dv.dp_query_test(d1_query, d2_query, plot=False, repeat_count=1000)

        d1_query = "SELECT Role, Segment, COUNT(UserId) AS UserCount, SUM(Usage) AS Usage FROM d1.d1 GROUP BY Role, Segment"
        d2_query = "SELECT Role, Segment, COUNT(UserId) AS UserCount, SUM(Usage) AS Usage FROM d2.d2 GROUP BY Role, Segment"
        dp_res, acc_res, utility_res, bias_res = dv.dp_groupby_query_test(d1_query, d2_query, plot=False, repeat_count=1000)

        # Powerset Test on SUM query
        query_str = "SELECT SUM(Usage) AS TotalUsage FROM "
        dp_res, acc_res, utility_res, bias_res = self.dp_powerset_test(query_str, plot=False, repeat_count=1000)

        # Yarrow Test
        # dataset_root = os.getenv('DATASET_ROOT', '/home/ankit/Documents/github/datasets/')
        # test_csv_path = dataset_root + 'data/PUMS_california_demographics_1000/data.csv'

        # dp_yarrow_mean_res = self.yarrow_test(test_csv_path, yarrow.dp_mean, 'income', float, epsilon=self.epsilon, minimum=0, maximum=100, num_records=1000)
        # dp_yarrow_var_res = self.yarrow_test(test_csv_path, yarrow.dp_variance, 'educ', int, epsilon=self.epsilon, minimum=0, maximum=12, num_records=1000)
        # dp_yarrow_moment_res = self.yarrow_test(test_csv_path, yarrow.dp_moment_raw, 'married', float, epsilon=.15, minimum=0, maximum=12, num_records=1000000, order = 3)
        # dp_yarrow_covariance_res = self.yarrow_test(test_csv_path, yarrow.dp_covariance, 'married', int, 'sex', int, epsilon=.15, minimum_x=0, maximum_x=1, minimum_y=0, maximum_y=1, num_records=1000)
        return dp_res, acc_res, utility_res, bias_res

if __name__ == "__main__":
    dv = DPVerification(dataset_size=1000)
    print(dv.main())
