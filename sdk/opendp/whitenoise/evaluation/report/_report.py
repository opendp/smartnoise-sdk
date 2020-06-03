import pandas as pd

class Report:
	"""
	Defines the fields used to set evaluation parameters
	and consumed by the evaluator
	"""
	def __init__(self):
		self.noisy_vals = []
		self.exact_val = []
		self.dim_cols = []
		self.num_cols = []