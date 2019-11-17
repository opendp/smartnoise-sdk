from abc import ABCMeta, abstractmethod
import math
import logging
from burdock.mechanisms.laplace import Laplace

logger = logging.getLogger(__name__)


class Statistic(object):

    __metaclass__ = ABCMeta

    def __init__(self, release, variable, accuracy, epsilon, interval):
        self._release = release
        self._variable = variable
        self._accuracy = accuracy
        self._epsilon = epsilon
        self._interval = interval

    @property
    def release(self):
        return self._release

    @property
    def variable(self):
        return self._variable

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def interval(self):
        return self._interval

    def as_dict(self):
        return {"release": self._release,
                "variable": self._variable,
                "accuracy": self._accuracy,
                "epsilon": self._epsilon,
                "interval": self._interval}


class Computer(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def release(self, dataset):
        pass

    @staticmethod
    def get_subclasses():
        subclasses = Computer.__subclasses__()
        # Return dictionary with key value pairs from subclass name to subclass
        return {subclass.__name__.lower(): subclass for subclass in subclasses}


class CountResult(Statistic):
    def __init__(self, count_release, variable, accuracy, epsilon, interval):
        super(CountResult, self).__init__(count_release, variable, accuracy, epsilon, interval)
        self._count_release = count_release


class Count(Computer):
    def __init__(self, column, epsilon):
        self._column = column
        self._epsilon = epsilon

    def _compute_accuracy(self, epsilon, stability=None, delta=10^-6, alpha=0.05):
        if stability:
            return 2 * math.log(2 / (alpha * delta)) / epsilon
        else:
            return 2 * math.log(1 / alpha) / epsilon

    def release(self, dataset):
        # get the column count
        num_obs = dataset.shape[0]
        # obfuscate the count
        sens = 2
        tau = 5
        counts = Laplace(self._epsilon, tau).count([num_obs])
        count_release = counts[0]

        # calculate accuracy from epsilon
        accuracy = self._compute_accuracy(self._epsilon)
        accuracy_bound = accuracy * num_obs
        mci = [num_obs - accuracy_bound, num_obs + accuracy_bound]
        return CountResult(count_release, self._column, accuracy, self._epsilon, mci)
