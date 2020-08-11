class PrivacyParams:
	"""
	Defines the fields used to set privacy parameters
    and consumed by the evaluator
	"""
	def __init__(self, epsilon = 1.0, delta = 1.0, sens = 1.0, conf = 0.95, budget = 1):
		self.epsilon = epsilon
		self.delta = delta
		self.sens = sens
		self.conf = conf
		self.t = budget