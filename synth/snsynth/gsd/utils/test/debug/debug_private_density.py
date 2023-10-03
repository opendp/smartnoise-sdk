import subprocess
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
# from snsynth.gsd import GSDSynthesizer
from snsynth.gsd.utils.dataset_jax import Dataset, Domain
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, _get_density_fn, get_quantiles
from snsynth.utils import cdp_rho
import matplotlib.pyplot as plt

from snsynth.gsd.utils.utils import get_sigma, _divide_privacy_budget
from snsynth.utils import gaussian_noise

def post_histogram_tree(hist_tree: dict):
    # normalize_hist_tree = {}

    H = hist_tree['tree_height']

    # First level must add up to 1
    hist0 = np.clip(np.array(hist_tree['stats'][0]), 0, 1)
    hist0 = hist0 / hist0.sum()
    hist_tree['stats'][0] = hist0

    for h in range(1, H):
        reg_hist = np.array(hist_tree['stats'][h])
        # normalized_hist = np.clip(np.array(hist_tree['stats'][h]), 0, 1)
        normalized_hist = np.array(hist_tree['stats'][h])
        hist_len = normalized_hist.shape[0]

        for i in range(0, hist_len, 2):
            parent_i = i // 2
            parent_stat = hist_tree['stats'][h-1][parent_i]
            assert parent_stat>=0
            child_hist = normalized_hist[i:i+2]

            if child_hist.sum() < 1e-9:
                normalized_hist[i:i+2] = parent_stat * np.ones(2) / 2
            else:
                child_hist = parent_stat * child_hist / child_hist.sum()
                child_hist_clipped = np.clip(child_hist, 0, 1)
                normalized_hist[i:i+2] = child_hist_clipped

        print(f'h={h}. hist.sum={normalized_hist.sum():.2f}\treg_hist = {reg_hist.sum():.2f}')
        hist_tree['stats'][h] = list(normalized_hist)

    return hist_tree

def naive_density(values, data_range, eps):
    N = values.shape[0]
    rho = cdp_rho(eps, 1e-9)
    sigma = get_sigma(rho, sensitivity=np.sqrt(2) / N)

    num_bins = 1000
    bin_edges = np.linspace(0, data_range, num_bins)
    hist = np.histogram(values, bins=bin_edges)[0] / values.shape[0]
    density = hist.cumsum()

    noisy_hist = hist + gaussian_noise(sigma=sigma, size=hist.shape[0])  # Add DP
    noisy_density = noisy_hist.cumsum()

    noisy_hist_clipped = np.clip(noisy_hist, 0, 1)
    noisy_density_clipped = noisy_hist_clipped.cumsum()

    ## Plot histogram
    fig, axs = plt.subplots(1, 3)
    fig.figsize=(30,10)
    axs[0].bar(bin_edges[1:], hist)
    axs[1].bar(bin_edges[1:], noisy_hist)
    axs[2].bar(bin_edges[1:], noisy_hist_clipped)
    # plt.yticks([])
    plt.show()

    plt.plot(bin_edges[1:], density, label='True Density')
    plt.plot(bin_edges[1:], noisy_density, label='Noisy Density')
    plt.plot(bin_edges[1:], noisy_density_clipped, label='Noisy-Clipped Density')

    plt.legend()
    plt.show()

np.random.seed(0)
data_range = 10000
meta_data = {'O1': {'type': 'int', 'size': data_range}}
values = np.concatenate((np.zeros(5000).astype(int),
                         np.random.randint(0, data_range, size=(1000,)),
                         # np.random.randint(1000, 2000, size=(200,)),
                         np.random.randint(5100, 5200, size=(3000,)),
                         np.random.randint(9100, 9200, size=(1000,))
                         ))
df = pd.DataFrame({'O1': values})
domain = Domain(config=meta_data)
data = Dataset(df, domain)
tree_height=10

bin_edges = _get_bin_edges(domain=data.domain, tree_height=tree_height)
_, _, marginals_info = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges, rho=None,
                                              store_marginal_stats=True,
                                              verbose=True)
_, _, threshold, density = get_quantiles(marginals_info[('O1',)], num_quantiles=100)

for eps in [0.1]:

    naive_density(data.df.values, data_range, eps)
    continue


    rho = cdp_rho(eps, 1e-9)
    _, _, priv_marginals_info = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges, rho=rho,
                                                  store_marginal_stats=True, verbose=True)

    q1, _, t1, priv_density1 = get_quantiles(priv_marginals_info[('O1',)], num_quantiles=100)

    normalized_marginal_tree = post_histogram_tree(priv_marginals_info[('O1',)])
    q2, _, t2, priv_density2 = get_quantiles(normalized_marginal_tree, num_quantiles=100)

    # assert densities == [0, 0.91, 0.96]
    density_error1 = np.abs(np.array(priv_density1) - np.array(density)).sum()
    density_error2 = np.abs(np.array(priv_density2) - np.array(density)).sum()

    print(f'eps={eps:.2f}: error1 = {density_error1:.5f}, error2 = {density_error2:.5f}')

    plt.subplot(2, 2, 1)
    plt.title('Priv Density Method 1')
    plt.plot(t1, priv_density1, label='Priv Density')
    plt.vlines(x=q1, ymin=0.5, ymax=1, colors='k', alpha=0.1)
    plt.plot(threshold, density, label='Original Density')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title('Priv Density Method 2')
    plt.vlines(x=q2, ymin=0.5, ymax=1, colors='k', alpha=0.1)
    plt.plot(t2, priv_density2, label='Priv Density')
    plt.plot(threshold, density, label='Original Density')
    plt.legend()

    plt.show()

    print(priv_density1[:4])
    print(f'quantiles1 = {len(q1)}')
    print(priv_density2[:4])
    print(f'quantiles2 = {len(q2)}')
    print()
