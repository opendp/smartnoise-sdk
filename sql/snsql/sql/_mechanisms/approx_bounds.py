import numpy as np
from opendp.mod import enable_features
from opendp.measurements import make_base_laplace

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

    vals = np.array(vals, dtype=np.float64)
    vals = vals[vals != np.inf]
    vals = vals[vals != -np.inf]
    vals = vals[~np.isnan(vals)]


    def edges(idx):
        if idx == bins:
            return (0.0, 1.0)
        elif idx > bins:
            return (2.0 ** (idx - bins - 1), 2.0 ** (idx - bins))
        elif idx == bins - 1:
            return (-1.0, -0.0)
        else:
            return (-1 * 2.0 ** np.abs(bins - idx - 1), -1 * 2.0 ** np.abs(bins - idx - 2))
        
    edge_list = [edges(idx) for idx in range(len(hist))]
    min_val = min([l for l, u in edge_list])
    max_val = max([u for l, u in edge_list]) - 1
    vals = np.clip(vals, min_val, max_val)

    # compute histograms
    for v in vals:
        bin = None
        for idx, (l, u) in enumerate(edge_list):
            if l <= v < u:
                bin = idx
                break
        if bin is None:
            bin = idx
        hist[bin] += 1

        # for testing
        l, u = edges(bin)

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