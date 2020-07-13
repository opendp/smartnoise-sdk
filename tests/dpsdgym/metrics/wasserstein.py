import numpy as np
import pandas as pd

def wasserstein_randomization(d1_large, d2_large, iters, downsample_size=100):
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
    From "REALLY USEFUL SYNTHETIC DATA
    A FRAMEWORK TO EVALUATE THE QUALITY OF
    DIFFERENTIALLY PRIVATE SYNTHETIC DATA"
    https://arxiv.org/pdf/2004.07740.pdf
    """
    # pip install pyemd
    # https://github.com/wmayner/pyemd
    from pyemd import emd_samples

    assert(len(d1_large) == len(d2_large))
    d1 = d1_large.sample(n=downsample_size)
    d2 = d2_large.sample(n=downsample_size)
    l_1 = len(d1)
    d3 = np.concatenate((d1,d2))
    distances = []
    for i in range(iters):
        np.random.shuffle(d3)
        n_1, n_2 = d3[:l_1], d3[l_1:]
        try:
            # PyEMD is sometimes memory intensive
            # Let's reduce bins if so
            dist = emd_samples(n_1, n_2, bins='auto')
        except MemoryError:
            dist = emd_samples(n_1, n_2, bins=10)
        distances.append(dist)
    
    # Safety check, to see if there are any valid 
    # measurements
    if len(distances) == 0:
        return -1 
    d_pd = pd.DataFrame(distances)
    print(d_pd.describe())
    return np.mean(np.array(distances))