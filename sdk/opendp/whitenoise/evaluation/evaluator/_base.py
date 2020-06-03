class EvaluatorInterface:
	"""
	Interface for various DP implementations to interface with tests available
	in evaluator. Evaluator tests for various properties of DP implementation
	like privacy, accuracy, utility and bias
	"""
	def evaluate(self, d1, d2, analysis, privacy_params, eval_params):
		"""
		Evaluates properties of black box DP implementations using 
			- DP Histogram Test
			- Accuracy Test
			- Utility Test
			- Bias Test
		
		d1 and d2 are neighboring datasets
		analysis is the DP implementation object
		Returns a metrics object
		"""