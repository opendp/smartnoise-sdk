from opendp.smartnoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.metrics._metrics import Metrics
from opendp.smartnoise.evaluation.evaluator._base import Evaluator
import numpy as np
from scipy import stats, spatial
import math
import matplotlib.pyplot as plt


class DPEvaluator(Evaluator):
    def _generate_histogram_neighbors(self, fD1, fD2, ep: EvaluatorParams):
        """
        Generate histograms given the vectors of repeated aggregation results
        applied on neighboring datasets
        """
        fD1 = np.asarray(fD1, dtype="float64")
        fD2 = np.asarray(fD2, dtype="float64")
        d = np.concatenate((fD1, fD2), axis=None)
        n = len(fD1)
        binlist = []
        minval = min(min(fD1), min(fD2))
        maxval = max(max(fD1), max(fD2))

        # Deciding bin width and bin list
        if ep.exact:
            binlist = np.linspace(minval, maxval, 2)
        elif ep.numbins > 0:
            binlist = np.linspace(minval, maxval, ep.numbins)
        elif ep.binsize == "auto":
            iqr = np.subtract(*np.percentile(d, [75, 25]))
            numerator = 2 * iqr if iqr > 0 else maxval - minval
            denominator = n ** (1.0 / 3)
            binwidth = numerator / denominator  # Freedman–Diaconis' choice
            ep.numbins = int(math.ceil((maxval - minval) / binwidth)) if maxval > minval else 20
            binlist = np.linspace(minval, maxval, ep.numbins)
        else:
            # Choose bin size of unity
            binlist = np.arange(np.floor(minval), np.ceil(maxval))

        # Calculating histograms of fD1 and fD2
        fD1hist, bin_edges = np.histogram(fD1, bins=binlist, density=False)
        fD2hist, bin_edges = np.histogram(fD2, bins=binlist, density=False)

        return fD1hist, fD2hist, bin_edges

    def _dp_test(
        self, d1hist, d2hist, binlist, d1size, d2size, ep: EvaluatorParams, pp: PrivacyParams
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
        if not ep.exact:
            num_buckets = binlist.size - 1
            critical_value = stats.norm.ppf(1 - (ep.alpha / 2 / num_buckets), loc=0.0, scale=1.0)
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
        bound_exceeded = np.any(
            np.logical_and(
                np.greater(d1hist, np.zeros(d1hist.size)), np.greater(d1lower, d2histupperbound)
            )
        ) or np.any(
            np.logical_and(
                np.greater(d2hist, np.zeros(d2hist.size)), np.greater(d2lower, d1histupperbound)
            )
        )

        return not bound_exceeded, d1histupperbound, d2histupperbound, d1lower, d2lower

    def _plot_histogram_neighbors(
        self,
        fD1,
        fD2,
        d1histupperbound,
        d2histupperbound,
        d1hist,
        d2hist,
        d1lower,
        d2lower,
        binlist,
        ep: EvaluatorParams,
    ):
        """
        Plot histograms given the vectors of repeated aggregation results
        applied on neighboring datasets
        """
        plt.figure(figsize=(15, 5))
        if ep.exact:
            ax = plt.subplot(1, 1, 1)
            ax.ticklabel_format(useOffset=False)
            plt.xlabel("Bin")
            plt.ylabel("Probability")
            plt.hist(fD1, width=0.2, alpha=0.5, ec="k", align="right", bins=1)
            plt.hist(fD2, width=0.2, alpha=0.5, ec="k", align="right", bins=1)
            ax.legend(["D1", "D2"], loc="upper right")
            return

        ax = plt.subplot(1, 2, 1)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel("Bin")
        plt.ylabel("Probability")
        if ep.bound:
            plt.bar(
                binlist[:-1],
                d2histupperbound,
                alpha=0.5,
                width=np.diff(binlist),
                ec="k",
                align="edge",
            )
            plt.bar(binlist[:-1], d1lower, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(["D1", "D2"], loc="upper right")
        else:
            plt.bar(binlist[:-1], d1hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d2hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(["D1", "D2"], loc="upper right")

        ax = plt.subplot(1, 2, 2)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel("Bin")
        plt.ylabel("Probability")
        if ep.bound:
            plt.bar(
                binlist[:-1],
                d1histupperbound,
                alpha=0.5,
                width=np.diff(binlist),
                ec="k",
                align="edge",
            )
            plt.bar(binlist[:-1], d2lower, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(["D2", "D1"], loc="upper right")
        else:
            plt.bar(binlist[:-1], d2hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.bar(binlist[:-1], d1hist, alpha=0.5, width=np.diff(binlist), ec="k", align="edge")
            plt.legend(["D2", "D1"], loc="upper right")
        plt.show()

    def wasserstein_distance(self, fD1, fD2):
        """
        Wasserstein Distance between responses of repeated algorithm on neighboring datasets
        """
        return stats.wasserstein_distance(fD1.astype(np.float), fD2.astype(np.float))

    def jensen_shannon_divergence(self, fD1, fD2):
        """
        Jensen Shannon Divergence between responses of repeated algorithm on neighboring datasets
        """
        return spatial.distance.jensenshannon(fD1.astype(np.float), fD2.astype(np.float))

    def kl_divergence(self, fD1, fD2):
        """
        KL Divergence between responses of repeated algorithm on neighboring datasets
        """
        return stats.entropy(fD1.astype(np.float), fD2.astype(np.float))

    def bias_test(self, fD1, fD_actual, sig_level):
        """
        1 sample t-test to check if difference in actual and noisy responses 
        is not statistically significant
        """
        diff = fD1 - fD_actual
        tset, pval = stats.ttest_1samp(diff, 0.0)
        return pval >= sig_level

    """
    Implement the Evaluator interface that takes in two neighboring datasets
    D1 and D2 and a privacy algorithm. Then runs the algorithm on the
    two datasets to find whether that algorithm adheres to the privacy promise.
    """

    def evaluate(
        self,
        d1: object,
        d2: object,
        pa: PrivacyAlgorithm,
        algorithm: object,
        pp: PrivacyParams,
        ep: EvaluatorParams,
    ) -> {str: Metrics}:
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
        pa.prepare(algorithm, pp, ep)
        key_metrics = {}
        try:  # whether the query is applicable to a DP test
            d1report = pa.release(
                d1
            )  # e.g. ValueError: Attempting to select a column not in a GROUP BY clause: dataset.dataset.UserId
            # e.g. '(sqlite3.OperationalError) a GROUP BY clause is required before HAVING\n[SQL: SELECT * FROM df_for_diffpriv1234 WHERE ( Usage != UserId AND UserId > "B" AND Usage <= Role AND Usage < "fUpNFG" ) HAVING UserId = 87 OR UserId <= Role]\n(Background on this error at: http://sqlalche.me/e/e3q8)'
        except Exception as e:
            metrics = Metrics()
            key_metrics["__key__"] = metrics
            metrics.dp_res = None
            metrics.error = str(type(e)) + ", " + str(e)
            return key_metrics
        if (
            "__key__" in d1report.res
            and isinstance(d1report.res["__key__"], str)
            and (d1report.res["__key__"] == "noisy_values_empty")
        ):
            metrics = Metrics()
            key_metrics["__key__"] = metrics
            metrics.dp_res = None
            metrics.error = d1report.res["__key__"]
            return key_metrics
        d2report = pa.release(d2)
        d1actual = pa.actual_release(d1)
        if (
            "__key__" in d1report.res
            and isinstance(d1actual.res["__key__"], str)
            and d1actual.res["__key__"].startswith("exact_value_error")
        ):
            metrics = Metrics()
            key_metrics["__key__"] = metrics
            metrics.dp_res = None
            metrics.error = d1actual.res["__key__"]
            return key_metrics
        if "__key__" in d1report.res and d1actual.res["__key__"] is None:
            metrics = Metrics()
            key_metrics["__key__"] = metrics
            metrics.dp_res = None
            metrics.error = "exact_value_is_none"
            return key_metrics
        for key in d1report.res.keys():
            metrics = Metrics()
            fD1, fD2 = np.array(d1report.res[key]), np.array(d2report.res[key])
            if not (fD1.dtype == np.int or fD1.dtype == np.float):
                metrics = Metrics()
                key_metrics["__key__"] = metrics
                metrics.dp_res = None
                metrics.error = "not_a_numeric_object"
                return key_metrics
            fD_actual = d1actual.res[key]
            d1hist, d2hist, bin_edges = self._generate_histogram_neighbors(fD1, fD2, ep)
            dp_res, d1histupperbound, d2histupperbound, d1lower, d2lower = self._dp_test(
                d1hist, d2hist, bin_edges, fD1.size, fD2.size, ep, pp
            )

            # Compute Metrics
            metrics.dp_res = dp_res
            metrics.wasserstein_distance = self.wasserstein_distance(fD1, fD2)
            metrics.jensen_shannon_divergence = self.jensen_shannon_divergence(fD1, fD2)
            metrics.kl_divergence = self.kl_divergence(fD1, fD2)
            metrics.mse = np.mean((fD1 - fD_actual) ** 2)
            metrics.msd = np.sum(fD1.astype(np.float) - fD_actual) / fD1.size
            metrics.std = np.std(fD1.astype(np.float))
            metrics.bias_res = self.bias_test(fD1, fD_actual, ep.sig_level)

            # Add key and metrics to final result
            key_metrics[key] = metrics

            # Break if only single key needs to be evaluated in the report
            if ep.eval_first_key:
                break

        return key_metrics
