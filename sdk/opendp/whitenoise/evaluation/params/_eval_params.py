class EvaluatorParams:
	"""
	Defines the fields used to set evaluation parameters
    and consumed by the evaluator
	"""
	def __init__(self, repeat_count=500):
		self.repeat_count = repeat_count
		self.numbins = 0
		self.binsize="auto"
		self.exact=False
		self.alpha=0.05
		self.bound = True