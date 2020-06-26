import pandas as pd
import numpy as np

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
	def __init__(self, df):
		self.res_df = df
		self.dim_cols = []
		self.num_cols = []

		for col in self.res_df:
			print(self.res_df[col].dtype)
			if(self.res_df[col].dtype != np.number):
				self.dim_cols.append(col)
			else:
				self.num_cols.append(col)
		
		if(len(self.dim_cols) == 0):
			self.dim_cols.append("__dim__")

		if(self.dim_cols[0] == "__dim__"):
			self.res_df[self.dim_cols[0]] = ["key"]*len(self.res_df)