class BenchmarkParams:
	"""
	Defines the fields used to set benchmarking parameters
    and consumed by the benchmarking API
    Algorithms are the list of DP algorithms that need to be benchmarked
	"""
	def __init__(self, algorithms):
		self.algorithms = []