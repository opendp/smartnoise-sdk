class LearnerInterface:
	"""
	Interface for smarter exploration of datasets and test queries 
	for finding DP property violations
	"""
	def create(self, setup_params):
		"""
		Multi-armed bandit approach: Setup the number of bandits
		or queries and query candidate pool. 
		Reinforcement Learning approach: setup seed query
		"""
		
	def notify(self, analysis, metrics, privacy_params):
		"""
		Tells the learner about the results of an analysis. 
		Returns null
		"""

	def propose(self):
		"""
		Asks the learner to propose a new analysis that optimizes
		the given objective metrics.  May be totally random.  
		Returns analysis object.
		"""