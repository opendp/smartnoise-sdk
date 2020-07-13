import numpy as np
import pandas as pd

def sra(R, S):
    """
    SRA can be thought of as the (empirical) probability of a
    comparison on the synthetic data being ”correct” (i.e. the same as
    the comparison would be on the real data).
    
    From "Measuring the quality of Synthetic data for use in competitions"
    https://arxiv.org/pdf/1806.11345.pdf

    (NOTE: SRA requires at least 2 accuracies per list to work)

    :param R: list of accuracies on models of real data
    :type R: list of floats
    :param S: list of accuracies on models of synthetic data
    :type S: list of floats
    :return: sra score
    :rtype: float
    """
    print(R, S)
    k = len(R)
    sum_I = 0
    for i in range(k):
        R_vals = np.array([R[i]-rj if i != k else None for k, rj in enumerate(R)])
        S_vals = np.array([S[i]-sj if i != k else None  for k, sj in enumerate(S)])
        I = (R_vals[R_vals != np.array(None)] * S_vals[S_vals != np.array(None)])
        I[I >= 0] = 1
        I[I < 0] = 0
        sum_I += I
    return np.sum((1 / (k * (k-1))) * sum_I)