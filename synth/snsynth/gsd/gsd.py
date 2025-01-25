import pandas as pd
import numpy as np

try:
    import jax
    import itertools
    import jax.numpy as jnp
except ImportError:
    print("Please install jax and flax:\nFor example you can run the install_cpu.sh or install_gpu.sh scripts.")

from jax.lib import xla_bridge
if xla_bridge.get_backend().platform == 'cpu':
    print("For GSD support, please install jax: pip install --upgrade  \"jax[cuda11_cudnn82]==0.4.6\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

from snsynth.base import Synthesizer
from snsynth.utils import cdp_rho, exponential_mechanism, gaussian_noise, powerset
from snsynth.transform import *
from snsynth.transform.type_map import TypeMap, ColumnTransformer
from snsynth.gsd.utils import Domain, Dataset
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, get_k_way_marginals
from snsynth.gsd.utils.generate_sync_data import generate, AVAILABLE_GENETIC_OPERATORS
from typing import Callable
from snsynth.transform.definitions import ColumnType

class GSDSynthesizer(Synthesizer):
    """
    Based on the paper: https://arxiv.org/abs/2306.03257
    """

    def __init__(self, epsilon=1., delta=1e-9, tree_height=20, verbose=False, *args, **kwargs):
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
            meta_data: dict = None,
            transformer=None,
            categorical_columns: list = None,
            ordinal_columns: list = None,
            continuous_columns: list = None,
            preprocessor_eps: float = 0.0,
            nullable=False,
            N_prime: int = None,
            early_stop_threshold: float = 0.0001,  # Increase this if you need to make GSD end sooner
            conditional_columns: list = (),   # Add extra dimension to the marginals. (Must be categorical)
            genetic_operators=(),
            marginals: list[tuple] = None,
            seed=None):

        self.data = self._get_data(data, transformer, categorical_columns, ordinal_columns,
                                   continuous_columns, preprocessor_eps,  nullable)
        self.dim = len(self.data.domain.attrs)
        self.N_prime = len(self.data.df) if N_prime is None else N_prime
        self.conditional_columns = conditional_columns

        # Get statistics
        self.true_stats, self.stat_fn, bin_edges = self.get_statistics(marginals)

        if self.verbose:
            print(f'Statistics size = {len(self.true_stats)}')

        if seed is None: seed = np.random.randint(0, 1e9)
        self.key = jax.random.PRNGKey(seed)

        # Define genetic strategy and parameters
        if len(genetic_operators) == 0:
            genetic_operators = AVAILABLE_GENETIC_OPERATORS

        domain = self.data.domain
        if bin_edges is not None:
            if self.verbose: print(f'Setting bin_edges for GSD.')
            domain.set_bin_edges(bin_edges)
        # Call genetic data generator
        self.sync_data = generate(self.key,
                                domain=domain,
                                N_prime=self.N_prime,
                                num_generations=50000000,
                                private_statistics=self.true_stats,
                                statistics_fn=self.stat_fn,
                                genetic_operators=genetic_operators,
                                early_stop_threshold=early_stop_threshold,
                                print_progress=self.verbose)

        data_list = self.get_values_as_list(self.data.domain, self.sync_data.df)
        self.sync_data_df = self._transformer.inverse_transform(data_list)

    def get_statistics(self, marginals: list[tuple] = None) -> (list, Callable, dict):
        data = self.data
        bin_edges = _get_bin_edges(domain=data.domain, tree_height=self.tree_height)

        priv_stats_2k, stat_fn_2k = _get_mixed_marginal_fn(data,
                                                           marginals_list=marginals,
                                                           bin_edges=bin_edges,
                                                           rho=self.rho,
                                                           verbose=self.verbose)
        if self.verbose:
            print(f'Statistics size = {len(priv_stats_2k)}')
        return priv_stats_2k, stat_fn_2k, None


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
                if col in domain.get_continuous_cols():
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

    def _get_data_old(self, data,
                  meta_data: dict=None,
                  transformer = None,
                  categorical_columns=None,
                  ordinal_columns=None,
                  continuous_columns=None,
                  preprocessor_eps: float = None,
                  nullable=False
                  ):

        columns = self.get_column_names(data)

        if meta_data is None:
            meta_data = {}
        if categorical_columns is None:
            categorical_columns = []
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []
        def add_unique(s , str_list: list):
            if s not in str_list:
                str_list.append(s)
        if meta_data is not None:
            for meta_col in columns:
                if  meta_col in  meta_data.keys() and 'type' in meta_data[meta_col].keys():
                    type = meta_data[meta_col]['type']
                    if type == 'string':
                        add_unique(meta_col, categorical_columns)
                    if type == 'int':
                        add_unique(meta_col, ordinal_columns)
                    if type == 'float':
                        add_unique(meta_col, continuous_columns)

        if len(continuous_columns) + len(ordinal_columns) + len(categorical_columns) == 0:
            inferred = TypeMap.infer_column_types(data)
            categorical_columns = inferred['categorical_columns']
            ordinal_columns = inferred['ordinal_columns']
            continuous_columns = inferred['continuous_columns']
            if not nullable:
                nullable = len(inferred['nullable_columns']) > 0

        mapped_columns = categorical_columns + ordinal_columns + continuous_columns

        assert len(mapped_columns) == len(columns), 'Column mismatch. Make sure that the meta_data configuration defines all columns.'

        if transformer is None:
            col_tranformers = []
            for col in columns:
                if col in categorical_columns:
                    t = LabelTransformer(nullable=nullable)
                    col_tranformers.append(t)
                elif col in ordinal_columns:
                    lower = meta_data[col]['lower'] if col in meta_data  and 'lower'in meta_data[col] else None
                    upper = meta_data[col]['upper'] if col in meta_data and 'upper'in meta_data[col]else None
                    t = OrdinalTransformer(lower=lower, upper=upper, nullable=nullable)
                    col_tranformers.append(t)
                elif col in continuous_columns:
                    lower = meta_data[col]['lower'] if col in meta_data and 'lower'in meta_data[col] else None
                    upper = meta_data[col]['upper'] if col in meta_data and 'upper'in meta_data[col]else None
                    t = MinMaxTransformer(lower=lower, upper=upper, nullable=nullable, negative=False)
                    col_tranformers.append(t)
            self._transformer = TableTransformer(col_tranformers)
        else:
            self._transformer = transformer

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

        config = {}
        for col_tt, col in zip(self._transformer.transformers, columns):
            col_tt: ColumnTransformer
            if col in categorical_columns:
                cat_size = col_tt.cardinality[0]
                config[col] = {'type': 'string', 'size': cat_size}
            if col in ordinal_columns:
                ord_size = col_tt.cardinality[0]
                config[col] = {'type': 'int', 'size': ord_size}
            if col in continuous_columns:
                config[col] = {'type': 'float', 'size': 1}

        domain = Domain(config)
        data = Dataset(pd.DataFrame(np.array(train_data), columns=columns), domain)
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.continuous_columns = continuous_columns

        return data

    def _get_data(self, data,
                  transformer=None,
                  categorical_columns=None,
                  ordinal_columns=None,
                  continuous_columns=None,
                  preprocessor_eps: float = None,
                  nullable=False
                  ):

        columns = self.get_column_names(data)

        if categorical_columns is None:
            categorical_columns = []
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []

        mapped_columns = categorical_columns + ordinal_columns + continuous_columns

        if (len(mapped_columns) == 0) and transformer is None:
            # If column types and transformer are not specified then infer them
            inferred = TypeMap.infer_column_types(data)
            categorical_columns = inferred['categorical_columns']
            ordinal_columns = inferred['ordinal_columns']
            continuous_columns = inferred['continuous_columns']
            if not nullable:
                nullable = len(inferred['nullable_columns']) > 0
            mapped_columns = categorical_columns + ordinal_columns + continuous_columns

        if transformer is None:
            # If transformer is not passed then use the column types to create one
            assert len(mapped_columns) == len(columns), 'Column mismatch. Make sure all columns.'

            col_tranformers = []
            for col in columns:
                if col in categorical_columns:
                    t = LabelTransformer(nullable=nullable)
                    col_tranformers.append(t)
                elif col in ordinal_columns:
                    lower = None
                    upper = None
                    t = OrdinalTransformer(lower=lower, upper=upper, nullable=nullable)
                    col_tranformers.append(t)
                elif col in continuous_columns:
                    lower = None
                    upper = None
                    t = MinMaxTransformer(lower=lower, upper=upper, nullable=nullable, negative=False)
                    col_tranformers.append(t)
            self._transformer = TableTransformer(col_tranformers)

        config = {}
        assert len(transformer.transformers) == len(columns)
        self._transformer = transformer
        for col_tt, col in zip(self._transformer.transformers, columns):
            if col_tt.output_type == ColumnType.CATEGORICAL:
                cat_size = col_tt.cardinality[0]
                config[col] = {'type': 'string', 'size': cat_size}
                categorical_columns.append(col)
            if col_tt.output_type == ColumnType.ORDINAL:
                ord_size = col_tt.cardinality[0]
                config[col] = {'type': 'int', 'size': ord_size}
                ordinal_columns.append(col)
            if col_tt.output_type == ColumnType.CONTINUOUS:
                config[col] = {'type': 'float', 'size': 1}
                continuous_columns.append(col)


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

        domain = Domain(config)
        data = Dataset(pd.DataFrame(np.array(train_data), columns=columns), domain)
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.continuous_columns = continuous_columns

        return data