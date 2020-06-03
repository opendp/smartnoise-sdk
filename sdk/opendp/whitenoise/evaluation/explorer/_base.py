class ExplorerInterface:
	"""
	DP evaluator can be invoked with various evaluation parameters
	For example, for a SQL analysis, we can pass various datasets and 
	queries to see if the evaluator metrics are successful. This interface
	helps provide capability to do brute force generation of neighboring 
	datasets.
	"""
	def evaluate_powerset(self, dataset):
		"""
		Explores powerset of a given dataset
		"""
		
	def generate_halton(self):
		"""
		Generate new datasets using halton sequence. Calls the powerset explore
		"""