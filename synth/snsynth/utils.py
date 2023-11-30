import itertools

import numpy as np
from scipy.special import softmax
from opendp.measurements import make_laplace, make_gaussian
from opendp.mod import enable_features
import opendp.prelude as dp

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
    input_domain = dp.atom_domain(T=float)
    input_metric =  dp.absolute_distance(T=float)
    meas = make_gaussian(input_domain, input_metric, sigma)
    if size is None:
        return meas(0.0)
    else:
        return [meas(0.0) for _ in range(size)]

def laplace_noise(scale, size=None):
    enable_features('floating-point', 'contrib')
    input_domain = dp.atom_domain(T=float)
    input_metric =  dp.absolute_distance(T=float)
    meas = make_laplace(input_domain, input_metric, scale)
    if size is None:
        return meas(0.0)
    else:
        return [meas(0.0) for _ in range(size)]

def cdp_rho(epsilon, delta, max_contrib=1):
    # return a rho that satisfies (epsilon, delta)
    budget = (epsilon, delta)
    enable_features('floating-point', 'contrib')
    input_domain = dp.atom_domain(T=float)
    input_metric =  dp.absolute_distance(T=float)
    def make_adp_gauss(scale):
        test_gauss = make_gaussian(input_domain, input_metric, scale)
        adp = dp.c.make_zCDP_to_approxDP(test_gauss)
        return dp.c.make_fix_delta(adp, delta=delta)
    discovered_scale = dp.binary_search_param(
        lambda s: make_adp_gauss(s),
        d_in=float(max_contrib),
        d_out=budget)
    g = make_gaussian(input_domain, input_metric, discovered_scale)
    return g.map(d_in=1.0)

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))
