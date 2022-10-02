from typing import List
from ._mechanisms import *
from enum import Enum
import numpy as np

class Stat(Enum):
    count = 1
    sum_int = 2
    sum_large_int = 3
    sum_float = 4
    sum_large_float = 5
    threshold = 6

class Mechanisms:
    def __init__(self):
        self.classes = {
            Mechanism.laplace: Laplace,
            Mechanism.geometric: DiscreteLaplace,
            Mechanism.discrete_laplace: DiscreteLaplace,
            Mechanism.discrete_gaussian: DiscreteGaussian
        }
        self.large = 1000
        self.map = {
            Stat.count: Mechanism.discrete_laplace,
            Stat.sum_int: Mechanism.discrete_laplace,
            Stat.sum_large_int: Mechanism.discrete_laplace,
            Stat.sum_float: Mechanism.laplace,
            Stat.threshold: Mechanism.discrete_laplace
        }
    def _get_stat(self, stat: str, t: str):
        if stat == 'threshold':
            return Stat.threshold
        elif stat == 'count':
            return Stat.count
        elif stat == 'sum' and t in ['float', 'int']:
            return Stat.sum_int if t == 'int' else Stat.sum_float
        else:
            return None
    def get_mechanism(self, sensitivity, stat: str, t: str):
        if sensitivity is np.inf:
            return Unbounded
        stat = self._get_stat(stat, t)
        if stat is None:
            return None
        if stat is Stat.sum_int:
            if sensitivity > self.large and Stat.sum_large_int in self.map:
                stat = Stat.sum_large_int
        elif stat is Stat.sum_float:
            if sensitivity > self.large and Stat.sum_large_float in self.map:
                stat = Stat.sum_large_float
        if stat not in self.map:
            raise ValueError(f"Unable to determine which mechanism to use for {stat}")
        mech = self.map[stat]
        return self.classes[mech]
    @property
    def safe(self):
        return [Mechanism.geometric]

class Privacy:
    """Privacy parameters.  The Privacy object is passed in when creating
    any private SQL connection, and applies to all queries executed against that
    connection.

    :param epsilon: The epsilon value for each statistic returned by the private SQL connection.
    :param delta: The delta value for each query processed by the private SQL connection.  Most counts and sums will use delta of 0, but dimension
        censoring and Gaussian mechanism require delta.  Set delta to something like
        1/n*sqrt(n), where n is the approximate number of rows in the data source.
    :param alphas: A list of floats representing desired accuracy bounds.  Only set this parameter if you plan
        to use execute_with_accuracy for row-based accuracy.  For simple column accuracy bounds, you can pass
        an alpha directly to get_simple_accuracy, which ignores these alphas.
    :param mechanisms: A property bag specifying which mechanisms to use for which
        types of statistics.  You will only set this parameter if you want to override
        default mechanism mapping.
    """
    def __init__(self, *ignore, epsilon:float=1.0, delta:float=10E-16, alphas:List[float]=[], mechanisms:Mechanisms=None):
        """Privacy params.
        
        """
        self.epsilon = epsilon
        self.delta = delta
        self.alphas = alphas
        self.mechanisms = mechanisms if mechanisms else Mechanisms()