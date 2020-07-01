import numpy as np
import pandas as pd

def wasserstein_randomization(d1, d2, iters):
    """
    Calculate wasserstein randomization test results
    "We propose a metric based on
    the idea of randomization inference (Basu, 1980; Fisher, 1935). 
    Each data point is randomly assigned to one of two
    data sets and the similarity of the resulting two distributions 
    is measured with the Wasserstein distance. Repeating this
    random assignment a great number of times (e.g. 100000 times) 
    provides an empirical approximation of the distances’
    null distribution. Similar to the pMSE ratio score we then 
    calculate the ratio of the measured Wasserstein distance and
    the median of the null distribution to get a Wasserstein distance 
    ratio score that is comparable across different attributes.
    Again a Wasserstein distance ratio score of 0 would indicate that 
    two marginal distributions are identical. Larger scores
    indicate greater differences between distributions."
    """
    from scipy.stats import wasserstein_distance
    import matplotlib.pyplot as plt
    # pip install pyemd
    # https://github.com/wmayner/pyemd
    from pyemd import emd_samples

    assert(len(d1) == len(d2))
    l_1 = len(d1)
    d3 = np.concatenate((d1,d2))
    distances = []
    for i in range(iters):
        np.random.shuffle(d3)
        n_1, n_2 = d3[:l_1], d3[l_1:]
        dist = emd_samples(n_1, n_2, bins='auto')
        distances.append(dist)
    plt.hist(distances, bins=25)
    plt.show()

    d_pd = pd.DataFrame(distances)
    print(d_pd.describe())

    """
    import numpy as np
    import pandas as pd

    d1 = pd.read_csv('nursery.csv')
    d2 = pd.read_csv('synthetic.csv')

    d1 = d1.drop(d1.columns[[0]], axis=1)
    d2 = d2.drop(d2.columns[[0]], axis=1)

    wasserstein_randomization(d1.to_numpy(), d2.to_numpy(), 1000)
    """