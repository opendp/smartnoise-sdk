# This file contains a list of tests that can be passed actual aggregates or result aggregates from a DP implementation
# It tries to use a sample dataset S and splits it randomly into two neighboring datasets D1 and D2
# Using these neighboring datasets, it applies the aggregate query repeatedly
# It tests the DP condition to let the DP implementer know whether repeated aggregate query results are not enough to re-identify D1 or D2 which differ by single individual
# i.e. passing (epsilon, delta) - DP condition
# If the definition is not passed, there is a bug or it is a by-design bug in case of passing actual aggregates

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import Aggregation as agg
from scipy import stats

class DPVerification:
    # Set the epsilon parameter of differential privacy
    def __init__(self, epsilon=1.0, dataset_size=10000):
        self.epsilon = epsilon
        self.dataset_size = dataset_size
        self.df = self.create_simulated_dataset()
        print("Loaded " + str(len(self.df)) + " records")
        self.N = len(self.df)
        self.delta = 1/(self.N * math.sqrt(self.N))

    def create_simulated_dataset(self):
        np.random.seed(1)
        userids = list(range(1, self.dataset_size+1))
        userids = ["A" + str(user) for user in userids]
        usage = np.random.geometric(p=0.5, size=self.dataset_size).tolist()
        df = pd.DataFrame(list(zip(userids, usage)), columns=['UserId', 'Usage'])
        return df

    # Generate dataframes that differ by a single record that is randomly chosen
    def generate_neighbors(self):
        if(self.N == 0):
            print("No records in dataframe to run the test")
            return None, None
        
        d1 = self.df
        drop_idx = np.random.choice(self.df.index, 1, replace=False)
        d2 = self.df.drop(drop_idx)
        print("Length of D1: ", len(d1), " Length of D2: ", len(d2))
        return d1, d2

    # If there is an aggregation function that we need to test, we need to apply it on neighboring datasets
    # This function applies the aggregation repeatedly to log results in two vectors that are then used for generating histogram
    # The histogram is then passed through the DP test
    def apply_aggregation_neighbors(self, f, args1, args2):
        fD1 = f(*args1)
        fD2 = f(*args2)

        print("Mean fD1: ", np.mean(fD1), " Stdev fD1: ", np.std(fD1), " Mean fD2: ", np.mean(fD2), " Stdev fD2: ", np.std(fD2))
        return fD1, fD2

    # Instead of applying function to dataframe, this'll pass a query through PrivSQL and get response
    # This way we can test actual SQLDP implementation
    def apply_query(self, d1, d2, agg_query):
        # To do
        return None

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
            numbins = math.ceil(maxval - minval) / binwidth
            binlist = np.linspace(minval, maxval, numbins)
        else:
            # Choose bin size of unity
            binlist = np.arange(np.floor(minval),np.ceil(maxval))
        
        # Calculating histograms of fD1 and fD2
        d1hist, bin_edges = np.histogram(d1, bins = binlist, density = False)
        print("Sum of frequencies in D1 Histogram: ", np.sum(d1hist))
        d2hist, bin_edges = np.histogram(d2, bins = binlist, density = False)
        print("Sum of frequencies in D2 Histogram: ", np.sum(d2hist))

        return d1hist, d2hist, bin_edges
    
    # Plot histograms given the vectors of repeated aggregation results applied on neighboring datasets
    def plot_histogram_neighbors(self, fD1, fD2, d1hist, d2hist, binlist, d1size, d2size, bound=True, exact=False):
        plt.figure(figsize=(15,6))
        if(exact):
            ax = plt.subplot(1, 1, 1)
            ax.ticklabel_format(useOffset=False)
            plt.xlabel('Bin')
            plt.ylabel('Probability')
            plt.hist(fD1, width=0.2, alpha=0.5, ec="k", align = "right", bins = 1)
            plt.hist(fD2, width=0.2, alpha=0.5, ec="k", align = "right", bins = 1)
            ax.legend(['D1', 'D2'], loc="upper right")
            return
        
        px, py, d1histupperbound, d2histupperbound, d1histbound, d2histbound, d1lower, d2lower = \
            self.get_bounded_histogram(d1hist, d2hist, binlist, d1size, d2size, exact = exact)
        
        ax = plt.subplot(1, 2, 1)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
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
        plt.ylabel('Frequency')
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
        return not bound_exceeded

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

    def aggtest(self, f, colname, numbins=0, binsize="auto", debug=False, plot=True, bound=True, exact=False):
        d1, d2 = self.generate_neighbors()
        
        fD1, fD2 = self.apply_aggregation_neighbors(f, (d1, colname), (d2, colname))
        d1size, d2size = fD1.size, fD2.size

        ks_res = self.ks_test(fD1, fD2)
        print("\nKS 2-sample Test Result: ", ks_res, "\n")
        
        #andderson_res = self.anderson_ksamp(fD1, fD2)
        #print("Anderson 2-sample Test Result: ", andderson_res, "\n")
        
        d1hist, d2hist, bin_edges = \
            self.generate_histogram_neighbors(fD1, fD2, numbins, binsize, exact=exact)
        
        #kl_res = self.kl_divergence(d1hist, d2hist)
        #print("\nKL-Divergence Test: ", kl_res, "\n")

        ws_res = 0.0
        if(not exact):
            ws_res = self.wasserstein_distance(d1hist, d2hist)
        
        print("Wasserstein Distance Test: ", ws_res, "\n")

        dp_res = False
        if(not exact):
            dp_res = self.dp_test(d1hist, d2hist, bin_edges, d1size, d2size, debug, exact=exact)
        print("DP Predicate Test:", dp_res, "\n")
        
        if(plot):
            self.plot_histogram_neighbors(fD1, fD2, d1hist, d2hist, bin_edges, d1size, d2size, bound, exact)
        return dp_res, ks_res, ws_res

    # Main method listing all the DP verification steps
    def main(self):
        # Load simulated data to pandas dataframe
        print("1. Generate neighboring datasets from dataframe")
        d1, d2 = self.generate_neighbors()
        print("2. Apply the same aggregation function on both neighboring dataframes repeatedly")
        ag = agg.Aggregation()
        fD1, fD2 = self.apply_aggregation_neighbors(ag.dp_count, (d1, 'UserId'), (d2, 'UserId'), 10)
        print("3. Generate Histogram")
        d1hist, d2hist, bin_edges, d1error, d2error = self.generate_histogram_neighbors(fD1, fD2)
        print("4. DP Verification")
        res = self.dp_test(d1hist, d2hist, bin_edges, True)
        return res

if __name__ == "__main__":
    dv = DPVerification()
    print(dv.main())