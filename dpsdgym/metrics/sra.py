import numpy as np
import pandas as pd

def sra(real, synth):
    """
    SRA can be thought of as the (empirical) probability of a
    comparison on the synthetic data being ”correct” (i.e. the same as
    the comparison would be on the real data).

    From "Measuring the quality of Synthetic data for use in competitions"
    https://arxiv.org/pdf/1806.11345.pdf

    (NOTE: SRA requires at least 2 accuracies per list to work)

    :param real: list of accuracies on models of real data
    :type real: list of floats
    :param synth: list of accuracies on models of synthetic data
    :type synth: list of floats
    :return: sra score
    :rtype: float
    """
    k = len(real)
    sum_I = 0
    for i in range(k):
        R_vals = np.array([real[i]-rj if i != k else None for k, rj in enumerate(real)])
        S_vals = np.array([synth[i]-sj if i != k else None  for k, sj in enumerate(synth)])
        I = (R_vals[R_vals != np.array(None)] * S_vals[S_vals != np.array(None)])
        I[I >= 0] = 1
        I[I < 0] = 0
        sum_I += I
    return np.sum((1 / (k * (k-1))) * sum_I)