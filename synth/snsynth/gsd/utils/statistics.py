from snsynth.utils import gaussian_noise
from jax.config import config; config.update("jax_enable_x64", True)
import numpy as np
import jax
import jax.numpy as jnp
import chex
import pandas as pd
from snsynth.gsd.utils import Dataset, Domain
import itertools
from snsynth.gsd.utils.utils import get_sigma, _divide_privacy_budget


def get_thresholds_categorical(data_df: pd.Series,
                               size,
                               rho: float = None):
    """
    TODO: Add a maximum size parameter for extremely large cardinality features
    """
    N = len(data_df)
    sigma = get_sigma(rho, sensitivity=np.sqrt(2) / N)
    values = data_df.values

    bin_edges = np.linspace(0, size, size + 1) - 0.001
    hist, b_edges = np.histogramdd(values, bins=[bin_edges])
    stats = hist.flatten() / N
    stats_noised = stats + gaussian_noise(sigma=sigma, size=hist.shape[0])

    thresholds = {0: {'stats': stats_noised, 'bins': bin_edges}, 'tree_height' : 1}
    return thresholds



def _get_stats_fn(k, query_params):
    these_queries = jnp.array(query_params, dtype=jnp.float64)

    def answer_fn(x_row: chex.Array, query_single: chex.Array):
        I = query_single[:k].astype(int)
        U = query_single[k:2 * k]
        L = query_single[2 * k:3 * k]
        t1 = (x_row[I] < U).astype(int)
        t2 = (x_row[I] >= L).astype(int)
        t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
        answers = jnp.prod(t3)
        return answers

    temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))

    def scan_fun(carry, x):
        return carry + temp_stat_fn(x, these_queries), None

    def stat_fn(X):
        out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
        stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
        stats = jnp.round(stats)
        return stats / X.shape[0]

    return stat_fn


def _get_density_fn(query_params):
    these_queries = jnp.array(query_params, dtype=jnp.float64)

    def answer_fn(x_row: chex.Array, query_single: chex.Array):
        I = query_single[0].astype(int)
        threshold = query_single[1].astype(float)
        answer = (x_row[I] < threshold).astype(int)
        return answer

    temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))

    def scan_fun(carry, x):
        return carry + temp_stat_fn(x, these_queries), None

    def stat_fn(X):
        out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
        stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
        stats = jnp.round(stats)
        return stats / X.shape[0]

    return stat_fn


def _get_query_params(data, indices: list, marginal_bin_edges: list) -> (list, list):
    # indices = [data.domain.get_attribute_index(feature) for feature in features]
    answer_vectors = []
    query_params = []
    values = data.to_numpy_np()
    N = values.shape[0]
    bin_indices = [np.arange(bins.shape[0])[1:] for bins in marginal_bin_edges]
    bin_positions = list(itertools.product(*bin_indices))
    hist, b_edges = np.histogramdd(values[:, indices], bins=marginal_bin_edges)
    stats = hist.flatten() / N
    k = len(marginal_bin_edges)
    for stat_id in np.arange(stats.shape[0]):
        bins_idx: tuple
        bins_idx = bin_positions[stat_id]
        lower = []
        upper = []
        for i in range(k):
            bin_id: int
            bin_id = bins_idx[i]
            lower_threshold = marginal_bin_edges[i][bin_id - 1]
            upper_threshold = marginal_bin_edges[i][bin_id]
            lower.append(lower_threshold)
            upper.append(upper_threshold)
        query_params.append(np.concatenate((np.array(indices).astype(int), np.array(upper), np.array(lower))))
        answer_vectors.append(stats[stat_id])
    return answer_vectors, query_params

####################################################################################################

def _get_bin_edges(domain, tree_height):
    bins = {}

    for col in domain.attrs:
        bins[col] = {}
        if domain.is_categorical(col):
            size = domain.size(col)
            bins[col] = {'tree_height': 1, 0: np.linspace(0, size, size + 1) - 0.001}

        if domain.is_ordinal(col):
            size = domain.size(col)
            ord_col_tree_height = min(int(np.log2(size)+1), tree_height)
            bins[col] = {'tree_height': ord_col_tree_height}
            for h in range(ord_col_tree_height):
                h_bins = 2**(h+1)
                bins[col][h] = np.linspace(0, size, h_bins + 1)
                bins[col][h][-1] += 1e-6

        if domain.is_continuous(col):
            bins[col] = {'tree_height': tree_height}
            for h in range(tree_height):
                h_bins = 2 ** (h + 1)
                bins[col][h] = np.linspace(0, 1, h_bins + 1)
                bins[col][h][-1] += 1e-6
    return bins


def get_quantiles(bins: dict, num_quantiles: int):
    tree_height = bins['tree_height']
    all_thresholds = bins['bins'][tree_height-1][0]

    num_thresholds = all_thresholds.shape[0]
    num_bins = int(np.log2(num_thresholds-1))
    threshold_density = []
    threshold = []
    for i, thres in enumerate(all_thresholds[:-1]):
        if i == 0: continue
        binarystring = bin(i)
        prefix_len = num_bins - (len(binarystring)-2)
        binarystring = prefix_len * "0" + binarystring[2:]
        sum = 0
        pos = 0
        for tree_level, b in enumerate(binarystring):
            if b == '1':
                sum += bins['stats'][tree_level][pos]
                pos = 2 * pos + 2
            else:
                pos = 2 * pos
        # sum = np.clip(sum, 0, 1)
        threshold.append(thres)
        threshold_density.append(sum)

    # Find quantiles
    quantile = 1 / num_quantiles
    last_density = 0
    approx_quantiles = [0]
    densities = [0]
    for j in range(len(threshold_density)-1):
        edge = threshold[j]
        stat_lo = threshold_density[j]
        stat_hi = threshold_density[j+1]
        if (stat_hi - last_density) > quantile:
            approx_quantiles.append(edge)
            densities.append(stat_lo)
            last_density = stat_lo

    return approx_quantiles, densities, threshold, threshold_density


def _get_height_edges(column_bin_edges: dict, h):
    max_level = column_bin_edges['tree_height'] - 1
    return column_bin_edges[min(max_level, h)]

def get_k_way_marginals(domain: Domain, k: int, conditional_column: list=()):

    features = domain.get_continuous_cols() + domain.get_ordinal_cols() + domain.get_categorical_cols()
    for c in conditional_column:
        features.remove(c)
    marginals_list = []
    for marginal in [list(idx) for idx in itertools.combinations(features, k)]:
        marginals_list.append(marginal)
    return marginals_list

def check_marginals_lits_size(marginals_list):
    # Marginals must have the same size
    for m in marginals_list:
        assert len(m) == len(marginals_list[0])


def _get_mixed_marginal_fn(data: Dataset,
                           marginals_list: list[tuple],
                           bin_edges: dict,
                           maximum_size: int = None,
                           rho: float = None,
                           output_query_params: bool = False,
                           store_marginal_stats: bool = False,
                           verbose=False):
    domain = data.domain
    if marginals_list is None:
        # Default statistics is the 2-way marginals
        marginals_list = get_k_way_marginals(domain, k=2)
    check_marginals_lits_size(marginals_list)

    total_marginals = len(marginals_list)
    k_total = len(marginals_list[0])

    # Divide privacy budget and query capacity
    N = len(data.df)
    rho_split = _divide_privacy_budget(rho, total_marginals)
    query_params_total = []
    private_stats_total = []

    marginal_info = {}
    for marginal in marginals_list:
        # cond_marginal = list(marginal) + list(conditional_column)

        num_col_tree_height = [bin_edges[col]['tree_height'] for col in marginal]
        max_height = max(num_col_tree_height)
        marginal_max_size = maximum_size // (total_marginals) if maximum_size else None
        # marginal_max_size = min(marginal_max_size, N) if marginal_max_size else None
        marginal_info[tuple(marginal)] = {'tree_height': max_height, 'bins': {}, 'stats': {}}

        rho_split_level = _divide_privacy_budget(rho_split, max_height)  # Divide budget between tree_height

        sigma = get_sigma(rho_split_level, sensitivity=np.sqrt(2) / N)
        if verbose:
            print('Marginal=', marginal, f'. Sigma={sigma:.4f}. Top.Level={max_height}. Max.Size={marginal_max_size}')

        # Get the answers and select the bins with most information
        # indices = domain.get_attribute_indices(marginal)
        indices = [domain.get_attribute_index(col_name) for col_name in marginal]

        marginal_stats = []
        marginal_params = []
        for L in range(max_height):
            kway_edges = [_get_height_edges(bin_edges[col_name], L) for col_name in marginal]
            stats, query_params = _get_query_params(data, indices, kway_edges)
            priv_stats = np.array(stats)
            # if sigma > 0: priv_stats = np.clip(priv_stats + gaussian_noise(sigma=sigma, size=len(stats)), 0, 1) # Add DP
            if sigma > 0: priv_stats = priv_stats + gaussian_noise(sigma=sigma, size=len(stats)) # Add DP

            if store_marginal_stats:
                marginal_info[tuple(marginal)]['bins'][L] = kway_edges
                marginal_info[tuple(marginal)]['stats'][L] = list(priv_stats)

            marginal_stats += list(priv_stats)
            marginal_params += query_params

        private_stats_total += marginal_stats
        query_params_total += marginal_params

    if verbose:
        print(f'\tTotal size={len(private_stats_total)}')

    if store_marginal_stats:
        return private_stats_total, _get_stats_fn(k_total, query_params_total), marginal_info
    if output_query_params:
        return private_stats_total, _get_stats_fn(k_total, query_params_total), k_total, query_params_total
    return private_stats_total, _get_stats_fn(k_total, query_params_total)



