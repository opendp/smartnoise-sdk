import pandas as pd

class Report:
	"""
	Defines the consistent schema of reported fields
	that aid evaluation of a black box DP implementation
	* res_df: It is a dataframe that contains repeated 
	analysis results across dimension and numerical
	columns. It could be exact or noisy based on the 
	parameter actual = False or True in analysis
	* dim_cols: List of columns that contain dimension
	strings
	* num_cols: List of columns that contain numerical
	DP results
	"""
	def __init__(self):
		self.res_df = pd.DataFrame()
		self.dim_cols = []
		self.num_cols = []