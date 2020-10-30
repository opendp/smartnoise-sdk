from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.report._report import Report
from abc import ABC, abstractmethod

class PrivacyAlgorithm(ABC):
	"""
	Interface for every differential privacy algorithm to implement
	This shall help define functions that'll allow it to be evaluated whether
	the DP histogram test passes or not for such implementations.
	"""
	@abstractmethod
	def prepare(self, algorithm : object, privacy_params : PrivacyParams, eval_params: EvaluatorParams):
		"""
		Loads and compiles the specified algorithm into a PrivacyAlgorithm instance
		An algorithm is domain specific and can be any object. For example,
		it can be a graph, some sort of script written in any language, or a SQL
		query. privacy_params are a shared format that is consumed by the evaluator
		"""
		pass

	@abstractmethod
	def release(self, dataset : object) -> Report:
		"""
		Return a single report using the previously loaded algorithm and
		privacy_params applied on loaded dataset. The report must follow
		a consistent format, and includes the outbound parameters such as
		accuracy needed by the evaluator.

		Returns reports as a in-memory map<key, vector<double>>
		Spec details of data structure: https://docs.google.com/document/d/1VtFp4w3TRgFv7jDSEVUdKNk4VTqPkpX2jcO7qQM1YB4/edit#heading=h.qczap2x5w84o
		"""
		pass

	@abstractmethod
	def actual_release(self, dataset : object) -> Report:
		"""
		Return a single report using the previously loaded algorithm and 
		exact non-private response. 
		
		Returns reports as a in-memory map<key, double>
		"""
		pass
