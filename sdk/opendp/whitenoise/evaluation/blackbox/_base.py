from opendp.whitenoise.evaluation.params._privacy_params import PrivacyParams
from opendp.whitenoise.evaluation.params._eval_params import EvaluatorParams
from opendp.whitenoise.evaluation.report._report import Report
from abc import ABC, abstractmethod

class BlackBoxPrivacyInterface(ABC):
	"""
	Interface for every black box differential privacy algorithm to implement
	This shall help define functions that'll allow it to be evaluated whether
	the DP histogram test passes or not for such implementations. 
	"""
	@abstractmethod
	def prepare(self, analysis : object, privacy_params : PrivacyParams):
		"""
		Loads and compiles the specified analysis into a BlackBoxPrivacy instance
		An analysis is domain specific and can be any object. For example, 
		it can be a graph, some sort of script written in any language, or a SQL
		query. privacy_params are a shared format that is consumed by the evaluator
		"""
		pass
	
	@abstractmethod
	def release(self, dataset : object, actual = False) -> Report:
		"""
		Return a single report using the previously loaded analysis and 
		privacy_params applied on loaded dataset. The report must follow 
		a consistent format, and includes the outbound parameters such as 
		accuracy needed by the evaluator.
		
		Returns reports as a in-memory Python tabular vector of arrays.
		"""
		pass