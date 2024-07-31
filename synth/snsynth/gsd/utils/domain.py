import chex
import jax.numpy as jnp
import jax
import numpy as np
import random
from enum import Enum


class DataType(Enum):
    CATEGORICAL = 'string'
    CONTINUOUS = 'float'
    ORDINAL = 'int'


class Domain:
    def __init__(self, config: dict, null_cols: list = (), bin_edges: dict = None):
        """ Construct a Domain object

        :param attrs: a list or tuple of attribute names
        :param shape: a list or tuple of domain sizes for each attribute
        """
        self.attrs = list(config.keys())
        self.config = config


        self.null_cols = null_cols
        self._is_col_null = self._set_null_cols(null_cols)

        # Edges of real-value features used for discretization.
        self._bin_edges = bin_edges

    def has_nulls(self, col):
        return self._is_col_null[col]

    def set_bin_edges(self, bin_edges):
        self._bin_edges = bin_edges

    def _set_null_cols(self, null_cols):
        is_col_null = {}
        for col in self.attrs:
            is_col_null[col] = col in null_cols
        return is_col_null

    def get_bin_edges(self, col):
        """ Returns: First argument is a boolean indicating if col has bin edges defined. """
        if self._bin_edges is None:
            return False, None
        if col not in self._bin_edges:
            return False, None
        return True, jnp.array(self._bin_edges[col])

    def project(self, attrs):
        """ project the domain onto a subset of attributes

        :param attrs: the attributes to project onto
        :return: the projected Domain object
        """
        # return the projected domain
        if type(attrs) is str:
            attrs = [attrs]
        # shape = tuple(self.config[a] for a in attrs)
        new_config = {}
        for a in attrs:
            new_config[a] = self.config[a]
        return Domain(new_config, null_cols=self.null_cols, bin_edges=self._bin_edges)

    def axes(self, attrs):
        """ return the axes tuple for the given attributes

        :param attrs: the attributes
        :return: a tuple with the corresponding axes
        """
        return tuple(self.attrs.index(a) for a in attrs)

    def transpose(self, attrs):
        """ reorder the attributes in the domain object """
        return self.project(attrs)

    def invert(self, attrs):
        """ returns the attributes in the domain not in the list """
        return [a for a in self.attrs if a not in attrs]

    def contains(self, other):
        """ determine if this domain contains another

        """
        return set(other.attrs) <= set(self.attrs)

    def __contains__(self, attr):
        return attr in self.attrs

    def __getitem__(self, a):
        """ return the size of an individual attribute
        :param a: the attribute
        """
        return self.config[a]

    def __iter__(self):
        """ iterator for the attributes in the domain """
        return self.attrs.__iter__()

    def __len__(self):
        return len(self.attrs)

    def __eq__(self, other):
        return self.config == other.config

    def get_continuous_cols(self):
        n_cols = []
        for c in self.attrs:
            if self.config[c]['type'] in ('float', 'continuous'):
                n_cols.append(c)
        return n_cols

    def get_ordinal_cols(self):
        n_cols = []
        for c in self.attrs:
            if self.config[c]['type'] in ('int', 'ordinal'):
                n_cols.append(c)
        return n_cols

    def get_categorical_cols(self):
        c_cols = []
        for c in self.attrs:
            if self.config[c]['type'] in ('string', 'categorical'):
                c_cols.append(c)
        return c_cols

    def is_continuous(self, c):
        return self.config[c]['type'] in ('float', 'continuous')

    def is_categorical(self, c):
        return self.config[c]['type'] in ('string', 'categorical')

    def is_ordinal(self, c):
        return self.config[c]['type'] in ('int', 'ordinal')

    def size(self, att):
        if self.is_continuous(att): return 1
        return self.config[att]['size']

    def get_attribute_indices(self, atts: list) -> chex.Array:
        indices = []
        for i, temp in enumerate(self.attrs):
            if temp not in atts:
                continue
            indices.append(i)
        return jnp.array(indices)

    def get_attribute_index(self, att: str) -> int:
        for i, temp in enumerate(self.attrs):
            if temp == att: return i
        return -1


    def get_sampler(self, col, samples):

        if self. config[col]['type'] in ('string', 'categorical'):
            return self.get_categorical_sampler_jax(col, samples)
        if self. config[col]['type'] in ('float', 'continuous'):
            return self.get_numerical_sampler_jax(col, samples)
        if self. config[col]['type'] in ('int', 'ordinal'):
            return self.get_ordinal_sampler_jax(col, samples)

    def nulls_fn(self):
        def nulls(rng: chex.PRNGKey, col_values: chex.Array, num_nulls: int):
            n = col_values.shape[0]
            null_idx = jax.random.randint(rng, minval=0, maxval=n, shape=(num_nulls,))
            col_values = col_values.astype(float).at[null_idx].set(jnp.nan)
            return col_values
        return nulls

    def get_categorical_sampler_jax(self, col_name, samples):
        size = self.size(col_name)
        def sampling_fn(rng: chex.PRNGKey):
            c = jax.random.randint(rng, shape=(samples,), minval=0, maxval=size)
            return c
        return sampling_fn

    def get_numerical_sampler_jax(self, col_name, samples):
        has_bin_edges, bin_edges = self.get_bin_edges(col_name)

        if has_bin_edges:
            def sampling_fn(rng: chex.PRNGKey):
                rng_bin_0, rng_bin_1 = jax.random.split(rng, 2)
                num_bins = bin_edges.shape[0]
                bin_pos = jax.random.randint(rng_bin_0, shape=(samples,), minval=1, maxval=num_bins)
                right_edge = bin_edges[bin_pos]
                left_edge = bin_edges[bin_pos - 1]
                u = jax.random.uniform(rng_bin_1, shape=(samples,))
                c = (right_edge - left_edge) * u + left_edge
                return c
        else:
            def sampling_fn(rng: chex.PRNGKey):
                c = jax.random.uniform(rng, shape=(samples,))
                return c
        return sampling_fn

    def get_ordinal_sampler_jax(self, col_name, samples):
        size = self.size(col_name)
        has_bin_edges, bin_edges = self.get_bin_edges(col_name)

        if has_bin_edges:
            def sampling_fn(rng: chex.PRNGKey):
                # 1) Sample a bin
                rng_bin_0, rng_bin_1 = jax.random.split(rng, 2)
                num_bins = bin_edges.shape[0]
                bin_pos = jax.random.randint(rng_bin_0, shape=(samples,), minval=1, maxval=num_bins)
                # 2) Sample a value inside the interval edges
                right_edge = jnp.ceil(bin_edges[bin_pos]).astype(int)
                left_edge = jnp.ceil(bin_edges[bin_pos - 1]).astype(int)
                c = jax.random.randint(rng_bin_1, minval=left_edge, maxval=right_edge, shape=(samples,))
                return c
        else:
            def sampling_fn(rng: chex.PRNGKey):
                return jax.random.randint(rng, shape=(samples,), minval=0, maxval=size)
        return sampling_fn


    def get_log_sizes(self):
        # Get the cardinality or number of bins of each column
        log_sizes = []
        for att in self.attrs:
            has_bin_edges, bin_edges = self.get_bin_edges(att)
            if has_bin_edges:
                num_bins = bin_edges.shape[0]
                log_num_bins = int(np.log(num_bins) + 1)
                log_sizes.append(log_num_bins)
            else:
                sz = int(np.log(self.size(att)) + 1)
                log_sizes.append(sz)
        return log_sizes

    def sample_columns_based_on_logsize(self):
        log_sizes = self.get_log_sizes()
        col_ids = []
        for i, (col, sz) in enumerate(zip(self.attrs, log_sizes)):
            for _ in range(sz):
                col_ids.append(i)
        random.shuffle(col_ids)
        return col_ids
