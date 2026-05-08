

from snsynth.utils import gaussian_noise, cdp_rho
from jax.config import config; config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
from snsynth.gsd.utils import Dataset, Domain
import itertools

from snsynth.gsd.utils.utils import get_sigma, _divide_privacy_budget


def get_private_quantiles(data_df: pd.Series, quantiles, tree_height: int = 20, rho: float = None,
                              range_size=1, verbose=False) -> (dict, list):
    """
    Assumes that minimum value is 0
    """
    min_bin_size = 1 / 32


    N = len(data_df)
    sigma = get_sigma(rho, sensitivity=np.sqrt(2 * tree_height) / N)
    values: np.array
    values = data_df.values
    hierarchical_thresholds = {}
    thresholds = [0, range_size/2, range_size]
    total_levels = tree_height
    for i in range(0, tree_height):
        level_split = False
        thresholds.sort()

        # Compute stats for this level based on 'thresholds'
        thresholds_temp = thresholds.copy()
        thresholds_temp[-1] += 1e-6
        stats = np.histogramdd(values, [thresholds_temp])[0].flatten() / N
        stats_priv = list(np.array(stats) + gaussian_noise(sigma, size=len(stats)))

        # Record stats and thresholds
        hierarchical_thresholds[i] = {'stats': stats_priv, 'bins': np.array(thresholds_temp)}

        for thres_id in range(1, len(thresholds)):
            left = thresholds[thres_id-1]
            right = thresholds[thres_id]
            interval_stat = stats_priv[thres_id-1]

            if (i <= 1) or (interval_stat > min_bin_size + 2*sigma):
                # Split bin if it contains enough information.
                mid = (right + left) / 2
                thresholds.append(mid)
                level_split = True

        if not level_split:
            # Stop splitting intervals
            total_levels = i
            break

    hierarchical_thresholds['levels'] = total_levels
    return hierarchical_thresholds, thresholds



if __name__ == "__main__":


    N = 1000
    df = pd.Series( (0.05 * np.random.randn(N)+0.5))

    real_q = np.quantile(df.values, q=np.linspace(0, 1, 32))
    _, q = get_private_quantiles(df, quantiles=32, rho=1)


    print(len(q))
    print(real_q)
    print(q)

    real_stats = np.histogramdd(df.values, [real_q])[0].flatten() / N
    priv_stats = np.histogramdd(df.values, [q])[0].flatten() / N

    print(real_stats.max())
    print(priv_stats.max())
    print()
