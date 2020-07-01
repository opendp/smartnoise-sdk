import numpy as np
import pandas as pd

def sra(R, S):
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