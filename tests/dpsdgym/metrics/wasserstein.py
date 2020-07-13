import numpy as np
import pandas as pd

def wasserstein_randomization(d1_large, d2_large, iters, downsample_size=100):
    """
    Combine synthetic and real data into two sets and randomly 
    divide the data into two new random sets. Check the wasserstein
    distance (earth movers distance) between these two new muddled sets.
    Use the measured wasserstein distance to compute the ratio between
    it and the median of the null distribution (earth movers distance on
    original set). A ratio of 0 would indicate that the two marginal 
    distributions are identical.

    From "REALLY USEFUL SYNTHETIC DATA
    A FRAMEWORK TO EVALUATE THE QUALITY OF
    DIFFERENTIALLY PRIVATE SYNTHETIC DATA"
    https://arxiv.org/pdf/2004.07740.pdf

    NOTE: We return the mean here. However, its best
    probably to analyze the distribution of the wasserstein score

    :param d1_large: real data
    :type d1_large: pandas DataFrame
    :param d2_large: fake data
    :type d2_large: pandas DataFrame
    :param iters: how many iterations to run the randomization
    :type iters: int
    :param downsample_size: we downsample the original datasets due
    to memory constraints
    :type downsample_size: int
    :return: wasserstein randomization mean
    :rtype: float
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