class Metrics:
	"""
	Defines the fields available in the metrics payload object
	"""
	def __init__(self):
		self.dp_res = False
		self.acc_res = False
		self.within_bounds = 0
		self.outside_bounds = 0
		self.utility_res = False
		self.bias_res = False
		self.msd = 0.0
