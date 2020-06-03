class Report:
	"""
	Defines the fields used to set evaluation parameters
	and consumed by the evaluator
	"""
	def __init__(self):
		self.noisy_vals = []
		self.exact_val = 0.0
		self.low = []
		self.high = []
		self.dim_cols = []
		self.num_cols = []