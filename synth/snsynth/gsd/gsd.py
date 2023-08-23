import pandas as pd
import numpy as np

print("For GSD support, please install jax: pip install --upgrade  \"jax[cuda11_cudnn82]==0.4.6\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

try:
    import jax
    import itertools
    import jax.numpy as jnp
except ImportError:
    print("Please install jax and flax:\nFor example you can run the install_cpu.sh script.")

from snsynth.base import Synthesizer
from snsynth.utils import cdp_rho, exponential_mechanism, gaussian_noise, powerset
from snsynth.transform import *
from snsynth.transform.type_map import TypeMap, ColumnTransformer
from snsynth.gsd.utils import Domain, Dataset
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, _get_density_fn, get_quantiles
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
            meta_data: dict = None,
            transformer=None,
            categorical_columns: list = None,
            ordinal_columns: list = None,
            continuous_columns: list = None,
            preprocessor_eps: float = 0.0,
            nullable=False,
            N_prime : int = None,
            genetic_operators=(),
            seed=None):


        self.data = self._get_data(data, meta_data, categorical_columns, ordinal_columns,
                                   continuous_columns,preprocessor_eps,  nullable)
        self.dim = len(self.data.domain.attrs)
        self.N_prime = len(self.data.df) if N_prime is None else N_prime

        self.true_stats, self.stat_fn = self.get_statistics(self.data)

        if self.verbose:
            print(f'Statistics size = {len(self.true_stats)}')

        if seed is None: seed = np.random.randint(0, 1e9)
        self.key = jax.random.PRNGKey(seed)

        # Define genetic strategy and parameters
        if len(genetic_operators)  == 0:
            genetic_operators = AVAILABLE_GENETIC_OPERATORS

        # Call genetic data generator
        self.sync_data = generate(self.key,
                                  domain=self.data.domain,
                                  N_prime=self.N_prime,
                                  num_generations=5000000,
                             private_statistics=self.true_stats,
                                  statistics_fn= self.stat_fn,
                                                                        genetic_operators=genetic_operators,
                                  print_progress=self.verbose)

        data_list = self.get_values_as_list(self.data.domain, self.sync_data.df)
        self.sync_data_df = self._transformer.inverse_transform(data_list)


    def get_statistics(self, data: Dataset):
        dim = len(data.domain.attrs)
        # Split privacy budget
        numeric_features = data.domain.get_continuous_cols() + data.domain.get_ordinal_cols()
        num_numeric_features = len(numeric_features)
        rho_1 = None
        rho_2 = None
        if dim == 1:
            rho_1 = self.rho
        elif dim > 1 and num_numeric_features == 0:
            rho_2 = self.rho
            if self.verbose:
                print(f'privacy budgets: Second moments = {rho_2:.6f}')
        elif dim > 1 and num_numeric_features > 0:
            num_2way_marginals = len(list(itertools.combinations(range(dim), 2)))
            rho_1 = self.rho * num_numeric_features / (num_numeric_features + num_2way_marginals)
            rho_2 = self.rho * num_2way_marginals / (num_numeric_features + num_2way_marginals)

            if self.verbose:
                print(f'privacy budgets: First moments = {rho_1:.6f}. Second moments = {rho_2:.6f}')

        tree_height  = self.tree_height
        bin_edges = _get_bin_edges(domain=data.domain, tree_height=tree_height)

        # Special case: 1-dimensional categorical data.
        if dim==1 and num_numeric_features == 0:
            priv_stats_1k, stat_fn_1k  = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges, rho=rho_1,
                                                                           features=data.domain.get_categorical_cols(),
                                                                           verbose=self.verbose)
            return priv_stats_1k, stat_fn_1k

        if num_numeric_features>0:
            # Compute thresholds for all numeric features using privacy budget rho_1
            _, _, marginals_info = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges, rho=rho_1,
                                                           store_marginal_stats=True,
                                                                           features=numeric_features,
                                                               verbose=self.verbose)

            # Compute density function
            density_fn_params = []
            priv_density = []
            for col in numeric_features:
                thres_q, den_q, thres_all, density_all = get_quantiles(marginals_info[(col,)], 200)
                # Save threshold for computing the density function
                col_id = data.domain.get_attribute_index(col)
                for q, d in zip(thres_q, den_q):
                    density_fn_params.append([col_id, q])
                    priv_density.append(d)

            # Obtain first moment statistics and function.
            priv_density_np = jnp.array(priv_density)
            density_fn = _get_density_fn(jnp.array(density_fn_params))

            # Special case: 1-dimensional numeric dataset
            if dim == 1:
                return priv_density_np, density_fn

            # Compute bins for 2-way marginals
            # We use a smaller tree height for 2-way marginals due to computational limits
            for col in numeric_features:
                num_quantiles = 50
                approx_quantiles, densities, _, _ = get_quantiles(marginals_info[(col,)], num_quantiles)
                # Split approximate quantile thresholds into levels
                approx_quantiles = np.array(approx_quantiles)
                col_tree_height = int(np.log2(approx_quantiles.shape[0]))
                if self.verbose:
                    print(f'{col:>10}.tree_height = {col_tree_height}. Thresholds={approx_quantiles.shape[0]}')
                temp_bins = {'tree_height': col_tree_height}
                col_size = data.domain.size(col)
                for h in range(col_tree_height):
                    if h < col_tree_height-1:
                        step = approx_quantiles.shape[0] // (2 ** (h+1))
                        p = np.arange(0, approx_quantiles.shape[0]+1, step=step)[:-1]
                        thresholds = np.concatenate((approx_quantiles[p], np.array([col_size + 1e-5])))
                    else:
                        thresholds = np.concatenate((approx_quantiles, np.array([col_size + 1e-5])))
                    temp_bins[h] = thresholds
                # Update bin edges of all numeric features
                bin_edges[col] = temp_bins

            # Get second moment statistics using privacy budget rho_2
            priv_stats_2k, stat_fn_2k = _get_mixed_marginal_fn(data, k=2, bin_edges=bin_edges, rho=rho_2, verbose=self.verbose)

            # Combine first and second moment statistics
            def stat_fn_all(X):
                stat_1k = density_fn(X)
                stat_2k = stat_fn_2k(X)
                return jnp.concatenate((stat_1k, stat_2k))

            # General case: d-dimensional mixed-type dataset:
            priv_stats_all = jnp.concatenate((jnp.array(priv_density_np), jnp.array(priv_stats_2k)))
            return priv_stats_all, stat_fn_all
        else:
            # Special case: d-dimensional categorical data only:
            priv_stats_2k, stat_fn_2k = _get_mixed_marginal_fn(data, k=2, bin_edges=bin_edges, rho=rho_2,
                                                               verbose=self.verbose)
            return priv_stats_2k, stat_fn_2k

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

    def _get_data(self, data,
                  meta_data: dict=None,
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
        for col_tt, col in zip(col_tranformers, columns):
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