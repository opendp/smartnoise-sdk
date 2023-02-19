import itertools

import numpy as np
from scipy.special import softmax
from opendp.measurements import make_base_laplace, make_base_gaussian
from opendp.mod import enable_features, binary_search_param
from opendp.combinators import make_zCDP_to_approxDP, make_fix_delta

prng = np.random


def exponential_mechanism(qualities, epsilon, sensitivity=1.0, base_measure=None):
    if isinstance(qualities, dict):
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

def gaussian_noise(sigma, size=None):
    enable_features('floating-point', 'contrib')
    meas = make_base_gaussian(sigma)
    if size is None:
        return meas(0.0)
    else:
        return [meas(0.0) for _ in range(size)]

def laplace_noise(scale, size=None):
    enable_features('floating-point', 'contrib')
    meas = make_base_laplace(scale)
    if size is None:
        return meas(0.0)
    else:
        return [meas(0.0) for _ in range(size)]

def cdp_rho(epsilon, delta):
    budget = (epsilon, delta)
    enable_features('floating-point', 'contrib')
    def make_fixed_approxDP_gaussian(scale):
        adp = make_zCDP_to_approxDP(make_base_gaussian(scale))
        return make_fix_delta(adp, delta=budget[1])
    scale = binary_search_param(
        make_fixed_approxDP_gaussian,
        d_in=1.0, d_out=budget, T=float)
    return make_base_gaussian(scale).map(1.)

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))
