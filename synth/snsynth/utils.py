import itertools

import numpy as np
from scipy.special import softmax

prng = np.random


def exponential_mechanism(qualities, epsilon, sensitivity=1.0, base_measure=None):
    if isinstance(qualities, dict):
        # import pandas as pd
        # print(pd.Series(list(qualities.values()), list(qualities.keys())).sort_values().tail())
        keys = list(qualities.keys())
        qualities = np.array([qualities[key] for key in keys])
        if base_measure is not None:
            base_measure = np.log([base_measure[key] for key in keys])
    else:
        qualities = np.array(qualities)
        keys = np.arange(qualities.size)

    """ Sample a candidate from the permute-and-flip mechanism """
    q = qualities - qualities.max()
    if base_measure is None:
        p = softmax(0.5 * epsilon / sensitivity * q)
    else:
        p = softmax(0.5 * epsilon / sensitivity * q + base_measure)

    return keys[prng.choice(p.size, p=p)]


def gaussian_noise(sigma, size):
    """ Generate iid Gaussian noise  of a given scale and size """
    return prng.normal(0, sigma, size)


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))
