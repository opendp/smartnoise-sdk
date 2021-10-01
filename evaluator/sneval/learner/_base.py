from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.metrics._metrics import Metrics


class Learner(ABC):
    """
	Interface for smarter exploration of datasets and test queries 
	for finding DP property violations
	"""

    @abstractmethod
    def create(self, setup_params: LearnerParams):
        """
		Multi-armed bandit approach: Setup the number of bandits
		or queries and query candidate pool. 
		Reinforcement Learning approach: setup seed query
		"""
        pass

    @abstractmethod
    def notify(self, algorithm: object, metrics: Metrics, privacy_params: PrivacyParams):
        """
		Tells the learner about the results of an algorithm. 
		Returns null
		"""
        pass

    @abstractmethod
    def propose(self):
        """
		Asks the learner to propose a new algorithm that optimizes
		the given objective metrics.  May be totally random.  
		Returns algorithm object.
		"""
        pass
