from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from opendp.whitenoise.evaluation.evaluator._base import Evaluator
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class DPEValuator(Evaluator):
    def _generate_histogram_neighbors(self, 
            fD1, 
            fD2, 
            ep : EvaluatorParams
        ):
        """
        Generate histograms given the vectors of repeated aggregation results
        applied on neighboring datasets
        """
        d = np.concatenate((fD1, fD2), axis=None)
        n = len(fD1)
        binlist = []
        minval = min(min(fD1), min(fD2))
        maxval = max(max(fD1), max(fD2))

		# Deciding bin width and bin list
        if(ep.exact):
            binlist = np.linspace(minval, maxval, 2)
        elif(ep.numbins > 0):
            binlist = np.linspace(minval, maxval, numbins)
        elif(ep.binsize == "auto"):
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
        fD1hist, bin_edges = np.histogram(fD1, bins = binlist, density = False)
        fD2hist, bin_edges = np.histogram(fD2, bins = binlist, density = False)

        return fD1hist, fD2hist, bin_edges

    def _dp_test(self, 
            d1hist, 
            d2hist, 
            binlist, 
            d1size, 
            d2size, 
            ep : EvaluatorParams,
            pp : PrivacyParams
		):
        """
        Differentially Private Predicate Test
        Check if histogram of fD1 values multiplied by e^epsilon and 
        summed by delta is bounding fD2 and vice versa
        Use the histogram results and create bounded histograms 
        to compare in DP test
        """
        d1_error_interval = 0.0
        d2_error_interval = 0.0
        # Lower and Upper bound
        if(not ep.exact):
            num_buckets = binlist.size - 1
            critical_value = stats.norm.ppf(1-(alpha / 2 / num_buckets), loc=0.0, scale=1.0)
            d1_error_interval = critical_value * math.sqrt(num_buckets / d1size) / 2
            d2_error_interval = critical_value * math.sqrt(num_buckets / d2size) / 2

        num_buckets = binlist.size - 1
        px = np.divide(d1hist, d1size)
        py = np.divide(d2hist, d2size)

        d1histbound = px * math.exp(pp.epsilon) + pp.delta
        d2histbound = py * math.exp(pp.epsilon) + pp.delta

        d1upper = np.power(np.sqrt(px * num_buckets) + d1_error_interval, 2) / num_buckets
        d2upper = np.power(np.sqrt(py * num_buckets) + d2_error_interval, 2) / num_buckets
        d1lower = np.power(np.sqrt(px * num_buckets) - d1_error_interval, 2) / num_buckets
        d2lower = np.power(np.sqrt(py * num_buckets) - d2_error_interval, 2) / num_buckets

        np.maximum(d1lower, 0.0, d1lower)
        np.maximum(d2lower, 0.0, d2lower)

        d1histupperbound = d1upper * math.exp(pp.epsilon) + pp.delta
        d2histupperbound = d2upper * math.exp(pp.epsilon) + pp.delta

        # Check if any of the bounds across the bins violate the relaxed DP condition
        bound_exceeded = np.any(np.logical_and(np.greater(d1hist, np.zeros(d1hist.size)), np.greater(d1lower, d2histupperbound))) or \
        np.any(np.logical_and(np.greater(d2hist, np.zeros(d2hist.size)), np.greater(d2lower, d1histupperbound)))

        return not bound_exceeded, d1hist, d2hist, bin_edges, d1histupperbound, d2histupperbound, d1lower, d2lower

    def _plot_histogram_neighbors(self, 
            fD1, 
            fD2, 
            d1histupperbound, 
            d2histupperbound, 
            d1hist, 
            d2hist, 
            d1lower, 
            d2lower, 
            binlist,
            ep : EvaluatorParams
        ):
        """
        Plot histograms given the vectors of repeated aggregation results 
        applied on neighboring datasets
        """
        plt.figure(figsize=(15,5))
        if(ep.exact):
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
        if(ep.bound):
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
        if(ep.bound):
            plt.bar(binlist[:-1], d1histupperbound, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d2lower, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(['D2', 'D1'], loc="upper right")
        else:
            plt.bar(binlist[:-1], d2hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d1hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(['D2', 'D1'], loc="upper right")
        plt.show()

    """
    Implement the Evaluator interface that takes in two neighboring datasets
    D1 and D2 and a privacy algorithm. Then runs the algorithm on the 
    two datasets to find whether that algorithm adheres to the privacy promise.
    """
    def evaluate(self, 
		d1 : object, 
		d2 : object, 
		algorithm : PrivacyAlgorithm, 
		privacy_params : PrivacyParams, 
		eval_params : EvaluatorParams) -> Metrics:
        """
		Evaluates properties of privacy algorithm DP implementations using 
			- DP Histogram Test
			- Accuracy Test
			- Utility Test
			- Bias Test
		
		d1 and d2 are neighboring datasets
		algorithm is the DP implementation object
		Returns a metrics object
		"""
        #TBD
        return