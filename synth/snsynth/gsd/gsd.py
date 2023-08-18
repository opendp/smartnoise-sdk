import sys
import pandas as pd
import numpy as np
import jax
import warnings

import jax.numpy as jnp
print("For GSD support, please install jax: pip install --upgrade  \"jax[cuda11_cudnn82]==0.4.6\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

from typing import Callable
from snsynth.base import Synthesizer
from snsynth.utils import cdp_rho, exponential_mechanism, gaussian_noise, powerset
from snsynth.transform import *
from snsynth.transform.type_map import TypeMap, ColumnTransformer
from snsynth.transform.definitions import ColumnType
from snsynth.gsd.utils import Domain, Dataset
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges
from snsynth.gsd.utils.genetic_strategy import EvoState, PopulationState, SDStrategy
from snsynth.gsd.utils.generate_sync_data import generate, AVAILABLE_GENETIC_OPERATORS


class GSDSynthesizer(Synthesizer):
    """
    Based on the paper: https://arxiv.org/abs/2306.03257
    """

    def __init__(self, epsilon=1., delta=1e-9, tree_height=10, verbose=False, *args, **kwargs):
        # GSDSynthesizer.__init__()
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.verbose = verbose

        self.rho = cdp_rho(epsilon=epsilon, delta=delta)
        self.tree_height = tree_height

    def fit(
            self,
            data, *ignore,
            transformer=None,
            categorical_columns: list = None,
            ordinal_columns: list = None,
            continuous_columns: list = None,
            data_bounds: dict = None,
            preprocessor_eps: float = 0.0,
            nullable=False,
            N_prime : int = None,
            genetic_operators=(),
            seed=None):


        self.data = self._get_data(data, preprocessor_eps, categorical_columns, ordinal_columns, continuous_columns,
                              data_bounds, nullable)
        self.dim = len(self.data.domain.attrs)
        self.N_prime = len(self.data.df) if N_prime is None else N_prime



        self.true_stats, self.stat_fn = self.get_statistics()

        if self.verbose:
            print(f'Statistics size = {len(self.true_stats)}')

        if seed is None: seed = np.random.randint(0, 1e9)
        self.key = jax.random.PRNGKey(seed)

        # Define genetic strategy and parameters
        if len(genetic_operators)  == 0:
            genetic_operators = AVAILABLE_GENETIC_OPERATORS

        # Call genetic data generator
        self.sync_data = generate(self.key,
                                  # self.g_strategy,
                                  domain=self.data.domain,
                                  N_prime=self.N_prime,
                                  num_generations=5000000,
                             private_statistics=self.true_stats,
                                  statistics_fn= self.stat_fn,
                                                                        genetic_operators=genetic_operators,
                                  print_progress=self.verbose)

        data_list = self.get_values_as_list(self.data.domain, self.sync_data.df)
        self.sync_data_df = self._transformer.inverse_transform(data_list)


    def get_quantiles(self, bins: dict, num_quantiles: int):
        approx_quantiles = []
        bin_data = 1/ num_quantiles
        all_thresholds = bins['bins'][self.tree_height-1][0]

        num_thresholds = all_thresholds.shape[0]
        num_bins = int(np.log2(num_thresholds-1))

        for i, tres in enumerate(all_thresholds-1):
            binarystring = bin(i)
            prefix_len = num_bins - (len(binarystring)-2)
            binarystring = prefix_len * "0" + binarystring[2:]
            print(binarystring)

        return approx_quantiles


    def get_statistics(self):
        ### TODO: Split privacy budget
        warnings.warn("Must split privacy budget")
        print()
        tree_height  = self.tree_height
        bin_edges = _get_bin_edges(domain=self.data.domain, tree_height=tree_height)

        priv_stats_1k, stat_fn_1k, marginals_info = _get_mixed_marginal_fn(self.data, k=1, bin_edges=bin_edges, rho=self.rho,
                                                           store_marginal_stats=True,
                                                               verbose=self.verbose)

        # Under development
        # for col in self.data.domain.get_numerical_cols():
        #     approx_quantiles = self.get_quantiles(marginals_info[(col,)], num_quantiles=32)
        if self.dim == 1:
            return priv_stats_1k, stat_fn_1k

        # Update bin edges so that continuous features have at most 32 bins.

        bin_edges = _get_bin_edges(domain=self.data.domain, tree_height=5)
        # Case d>1
        priv_stats_2k, stat_fn_2k = _get_mixed_marginal_fn(self.data, k=2, bin_edges=bin_edges, rho=self.rho, verbose=self.verbose)


        priv_stats_all = jnp.concatenate((jnp.array(priv_stats_1k), jnp.array(priv_stats_2k)))
        def stat_fn_all(X):
            stat_1k = stat_fn_1k(X)
            stat_2k = stat_fn_2k(X)
            return jnp.concatenate((stat_1k, stat_2k))
        return priv_stats_all, stat_fn_all
        # return priv_stats_2k, stat_fn_2k



    def get_values_as_list(self, domain: Domain, df: pd.DataFrame):
        data_as_list = []
        for i, row in df.iterrows():
            row_list = []
            for j, col in enumerate(domain.attrs):
                value= row[col]
                if col in domain.get_categorical_cols():
                    row_list.append(int(value))
                if col in domain.get_ordinal_cols():
                    row_list.append(int(value))
                if col in domain.get_numerical_cols():
                    row_list.append(float(value))
            data_as_list.append(row_list)
        return data_as_list



    def sample(self, samples=None):
        if samples is None:
            return self.sync_data_df

        data = self.sync_data_df.sample(n=samples, replace=(samples > self.N_prime))
        return data

    @staticmethod
    def get_column_names(data):
        if isinstance(data, pd.DataFrame):
            return data.columns
        elif isinstance(data, np.ndarray):
            return list(range(len(data[0])))
        elif isinstance(data, list):
            return list(range(len(data[0])))

    def _get_data(self, data,
                  preprocessor_eps,
                  categorical_columns,
                  ordinal_columns,
                  continuous_columns,
                  data_bounds: dict,
                  nullable=False
                  ):

        if data_bounds is None:
            data_bounds = {}
        if categorical_columns is None:
            categorical_columns = []
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []

        if len(continuous_columns) + len(ordinal_columns) + len(categorical_columns) == 0:
            inferred = TypeMap.infer_column_types(data)
            categorical_columns = inferred['categorical_columns']
            ordinal_columns = inferred['ordinal_columns']
            continuous_columns = inferred['continuous_columns']
            if not nullable:
                nullable = len(inferred['nullable_columns']) > 0

        # columns = categorical_columns + ordinal_columns + continuous_columns
        columns = self.get_column_names(data)
        col_tranformers = []
        config = {}
        for col in columns:
            if col in categorical_columns:
                t = LabelTransformer(nullable=nullable)
                col_tranformers.append(t)
            elif col in ordinal_columns:
                t = LabelTransformer(nullable=nullable)
                col_tranformers.append(t)
            elif col in continuous_columns:
                lower = None
                upper = None
                if col in data_bounds:
                    lower = data_bounds[col]['lower']
                    upper = data_bounds[col]['upper']
                t = MinMaxTransformer(lower=lower, upper=upper, nullable=nullable, negative=False)
                col_tranformers.append(t)
        self._transformer = TableTransformer(col_tranformers)
        train_data = self._get_train_data(
            data,
            style='NA',
            transformer=self._transformer,
            nullable=nullable,
            preprocessor_eps=preprocessor_eps,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns
            )

        for col_tt, col in zip(col_tranformers, columns):
            col_tt: ColumnTransformer
            if col in categorical_columns:
                cat_size = col_tt.cardinality[0]
                config[col] = {'type': 'categorical', 'size': cat_size}
            if col in ordinal_columns:
                ord_size = col_tt.cardinality[0]
                config[col] = {'type': 'ordinal', 'size': ord_size}
            if col in continuous_columns:
                config[col] = {'type': 'numerical', 'size': 1}

        domain = Domain(config)
        data = Dataset(pd.DataFrame(np.array(train_data), columns=columns), domain)
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.continuous_columns = continuous_columns

        return data


if __name__ == "__main__":
    # DEBUG:
    columns = ['R1', 'R2']
    col_bounds = {'R1': {'lower': 0, 'upper': 1}, 'R2': {'lower': 0, 'upper': 1}}
    N = 10
    data_array = np.column_stack([
        0.37 * np.ones(N),
        np.concatenate([0.64 * np.ones(N // 2), np.zeros(N // 2)]),
    ])
    df = pd.DataFrame(data_array, columns=columns)

    # Since we are passing the data bounds, we do not need to provide privacy budget for preprocessing.
    synth = GSDSynthesizer(1000000.0, 1e-5,
                           tree_height=4,
                           verbose=True)
    synth.fit(df,  continuous_columns=columns, data_bounds=col_bounds)

    max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
    assert max_error < 0.00