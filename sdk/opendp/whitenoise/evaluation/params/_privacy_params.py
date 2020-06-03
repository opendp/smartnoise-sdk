class PrivacyParams:
	"""
	Defines the fields used to set privacy parameters
    and consumed by the evaluator
	"""
	def __init__(self):
		self.epsilon = 1.0
		self.delta = 1.0
		self.conf = 0.95