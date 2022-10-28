import numpy as np
from opendp.mod import enable_features
from opendp.measurements import make_base_laplace

def quantile(vals, alpha, epsilon, lower, upper):
    """Estimate the quantile.
    from: http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf

    :param vals: A list of values.  Must be numeric.
    :param alpha: The quantile to estimate, between 0.0 and 1.0.  
    For example, 0.5 is the median.
    :param epsilon: The privacy budget to spend estimating the quantile.
    :param lower: A bounding parameter.  The quantile will be estimated only for values
    greater than or equal to this bound.
    :param upper: A bounding parameter.  The quantile will be estimated only for values
    less than or equal to this bound.
    :return: The estimated quantile.

    .. code-block:: python

        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        median = quantile(vals, 0.5, 0.1, 0, 100)
    """
    k = len(vals)
    vals = [lower if v < lower else upper if v > upper else v for v in vals]
    vals = sorted(vals)
    Z = [lower] + vals + [upper]
    Z = [-lower + v for v in Z]  # shift right to be 0 bounded
    y = [
        (Z[i + 1] - Z[i]) * np.exp(-epsilon * np.abs(i - alpha * k))
        for i in range(len(Z) - 1)
    ]
    y_sum = sum(y)
    p = [v / y_sum for v in y]
    idx = np.random.choice(range(k + 1), 1, False, p)[0]
    v = np.random.uniform(Z[idx], Z[idx + 1])
    return v + lower

def approx_bounds(vals, epsilon):
    """Estimate the minimium and maximum values of a list of values.
    from: https://desfontain.es/thesis/Usability.html#usability-u-ding-

    :param vals: A list of values.  Must be numeric.
    :param epsilon: The privacy budget to spend estimating the bounds.
    :return: A tuple of the estimated minimum and maximum values.

    .. code-block:: python
    
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lower, upper = approx_bounds(vals, 0.1)
    """
    bins = 64
    hist = [0.0] * bins * 2

    def edges(idx):
        if idx == bins:
            return (0.0, 1.0)
        elif idx > bins:
            return (2.0 ** (idx - bins - 1), 2.0 ** (idx - bins))
        elif idx == bins - 1:
            return (-1.0, -0.0)
        else:
            return (-1 * 2.0 ** np.abs(bins - idx - 1), -1 * 2.0 ** np.abs(bins - idx - 2))

    # compute histograms
    for v in vals:
        if v >= 0 and v < 1.0:
            bin = bins
            hist[bin] += 1
        elif v >= 1.0:
            bin = int(np.trunc(np.log2(v))) + bins + 1
            if bin < len(hist):
                hist[bin] += 1
        elif v < 0 and v >= -1.0:
            bin = bins - 1
            hist[bin] += 1
        else:
            bin = bins - int(np.trunc(np.log2(-v + 1))) - 1
            if bin > 0:
                hist[bin] += 1

        # for testing
        l, u = edges(bin)
        assert(l <= v < u)

    enable_features('floating-point', 'contrib')
    discovered_scale = 1.0 / epsilon

    meas = make_base_laplace(discovered_scale)
    hist = [meas(v) for v in hist]
    n_bins = len(hist)

    failure_prob = 10E-9
    highest_failure_prob = 1/(n_bins * 2)

    exceeds = []
    while len(exceeds) < 1 and failure_prob <= highest_failure_prob:
        p = 1 - failure_prob
        K = - np.log(2 - 2 * p ** (1 / (n_bins- 1))) / epsilon
        exceeds = [idx for idx, v in enumerate(hist) if v > K]
        failure_prob *= 10

    if len(exceeds) == 0:
        return (None, None)

    lower, upper = min(exceeds), max(exceeds)
    ll, _ = edges(lower)
    _, uu = edges(upper)
    return (float(ll), float(uu))