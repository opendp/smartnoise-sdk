from typing import List
from ._mechanisms import *
from enum import Enum

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
            Mechanism.geometric: Geometric,
            Mechanism.gaussian: Gaussian
        }
        self.large = 1000
        self.map = {
            Stat.count: Mechanism.geometric,
            Stat.sum_int: Mechanism.geometric,
            Stat.sum_large_int: Mechanism.laplace,
            Stat.sum_float: Mechanism.laplace,
            Stat.threshold: Mechanism.laplace
        }
    def get_mechanism(self, stat, sensitivity):
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

class Privacy:
    def __init__(self, *ignore, epsilon:float=1.0, delta:float=10E-16, alphas:List[float]=[], neighboring:str="addremove"):
        """Privacy parameters.  Values are keyword-only.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.alphas = alphas
        self.mechanisms = Mechanisms()
        if neighboring not in ["addremove", "substitute"]:
            raise ValueError("Neighboring definition must be 'addremove' or 'substitute'")
        self.neighboring = neighboring