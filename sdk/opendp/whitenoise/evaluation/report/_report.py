import pandas as pd
import numpy as np

class Report:
	"""
	Defines the consistent schema of reported fields
	that aid evaluation of a DP algorithm implementation
	res: It is a map<key, vector<double>> i.e. {key string, list of DP numerical results} 
	that contains repeated algorithm results across dimension key and numerical
	DP noisy results. 
	If exact is set to true in release method, then it returns {key string, actual result}
	Spec: https://docs.google.com/document/d/1VtFp4w3TRgFv7jDSEVUdKNk4VTqPkpX2jcO7qQM1YB4/edit#
	"""
	def __init__(self, res):
		self.res = res