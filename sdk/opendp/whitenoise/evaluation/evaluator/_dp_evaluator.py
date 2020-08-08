from opendp.whitenoise.evaluation.privacyalgorithm._base import PrivacyAlgorithm
from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.metrics._metrics import Metrics
from opendp.whitenoise.evaluation.evaluator._base import Evaluator

class DPEValuator(Evaluator):
    def wasserstein_distance(self, d1hist, d2hist):
        """
        Wasserstein Distance between histograms of repeated algorithm on neighboring datasets
        """
        return stats.wasserstein_distance(d1hist, d2hist)

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
        metrics = Metrics()
        pa.prepare(algorithm, pp, ep)
        d1report = pa.release(d1)
        d2report = pa.release(d2)
        firstkey = list(d1report.res.keys())[0]

        fD1, fD2 = np.array(d1report.res[firstkey]), np.array(d2report.res[firstkey])

        d1hist, d2hist, bin_edges = self._generate_histogram_neighbors(fD1, fD2, ep)
        dp_res, d1histupperbound, d2histupperbound, d1lower, d2lower = \
            self._dp_test(d1hist, d2hist, bin_edges, fD1.size, fD2.size, ep, pp)
        
        metrics.dp_res = dp_res
        metrics.wasserstein_distance = self.wasserstein_distance(d1hist, d2hist)
        return metrics