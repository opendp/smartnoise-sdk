import json
import jax.nn
import numpy as np
import jax.numpy as jnp
from jax import random
import pandas as pd
from snsynth.gsd.utils import Domain


class Dataset:
    def __init__(self, df, domain: Domain):
        """ create a Dataset object """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        self.domain = domain
        self.df = df.loc[:, domain.attrs]

    def __len__(self):
        return len(self.df)

    @staticmethod
    def synthetic_jax_rng(domain: Domain, N, rng, null_values: float=0.0):
        """
        Generate synthetic data conforming to the given domain
        :param domain: The domain object
        :param N: the number of individuals
        """
        d = len(domain.attrs)
        rng0, rng1 = jax.random.split(rng, 2)
        rng_split = jax.random.split(rng0, d)
        num_nulls = int(N * null_values)
        arr = []
        for (rng_temp, att) in zip(rng_split, domain.attrs):
            rng_temp0, rng_temp1 = jax.random.split(rng_temp, 2)
            c = None
            if domain.type(att) == 'categorical':
                c = jax.random.randint(rng_temp0, shape=(N,), minval=0, maxval=domain.size(att))
            elif domain.type(att) == 'ordinal':
                has_bin_edges, bin_edges = domain.get_bin_edges(att)
                if has_bin_edges:
                    # 1) Sample a bin
                    rng_bin_0, rng_bin_1 = jax.random.split(rng_temp0, 2)
                    num_bins = bin_edges.shape[0]
                    bin_pos = jax.random.randint(rng_bin_0, shape=(N,), minval=1, maxval=num_bins)
                    right_edge = jnp.ceil(bin_edges[bin_pos]).astype(int)
                    left_edge = jnp.ceil(bin_edges[bin_pos - 1]).astype(int)
                    c = jax.random.randint(rng_bin_1, minval=left_edge, maxval=right_edge, shape=(N,))
                    # u = jax.random.uniform(rng_bin_1, shape=(N,))
                    # c = (right_edge - left_edge) * u + left_edge
                else:
                    c = jax.random.randint(rng_temp0, shape=(N,), minval=0, maxval=domain.size(att))
            elif domain.type(att) == 'numerical':
                has_bin_edges, bin_edges = domain.get_bin_edges(att)
                if has_bin_edges:
                    rng_bin_0, rng_bin_1 = jax.random.split(rng_temp0, 2)
                    num_bins = bin_edges.shape[0]
                    bin_pos = jax.random.randint(rng_bin_0, shape=(N,), minval=1, maxval=num_bins)
                    right_edge = bin_edges[bin_pos]
                    left_edge = bin_edges[bin_pos-1]
                    u = jax.random.uniform(rng_bin_1, shape=(N, ))
                    c = (right_edge - left_edge) * u + left_edge

                else:
                    c = jax.random.uniform(rng_temp0, shape=(N, ))

            if domain.has_nulls(att):
                null_idx = jax.random.randint(rng_temp1, minval=0, maxval=N, shape=(num_nulls,))
                c = c.astype(float).at[null_idx].set(jnp.nan)
            arr.append(c)


        values = jnp.array(arr).T
        return values

    @staticmethod
    def synthetic_rng(domain: Domain, N, rng, null_values: float=0.0):
        """ Generate synthetic data conforming to the given domain """
        arr = [rng.uniform(size=N) if domain.type(att) == 'numerical'
                else rng.integers(low=0, high=domain.size(att), size=N).astype(float) for att in domain.attrs]

        values = np.array(arr).T
        temp = rng.uniform(low=0, high=1, size=values.shape) < null_values
        values[temp] = np.nan

        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def synthetic(domain, N, seed: int, null_values: float=0.0):
        rng = np.random.default_rng(seed)
        return Dataset.synthetic_rng(domain, N, rng, null_values)

    @staticmethod
    def load(path, domain):
        """ Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config)
        return Dataset(df, domain)
    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:,cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    # def datavector(self, flatten=True, weights=None, density=False):
    #     """ return the database in vector-of-counts form """
    #     bins = [range(n+1) for n in self.domain.shape]
    #     ans = np.histogramdd(self.df.values, bins, weights=weights, density=density)[0]
    #     return ans.flatten() if flatten else ans

    # def datavector_jax(self, flatten=True, weights=None, density=False):
    #     """ return the database in vector-of-counts form """
    #     # bins = jnp.array([range(n+1) for n in self.domain.shape])
    #     bins = [jnp.array(list(range(n+1))) for n in self.domain.shape]
    #     # bins = jnp.array([(n+1) for n in self.domain.shape])
    #     ans = jnp.histogramdd(self.df.values, bins, weights=weights, density=density)[0]
    #     return ans.flatten() if flatten else ans

    def sample(self, p=None, n=None, replace=False, seed=None):
        subsample = None
        if p is not None:
            subsample = self.df.sample(frac=p, replace=replace, random_state=seed)
        elif n is not None:
            subsample = self.df.sample(n=n, replace=replace, random_state=seed)
        return Dataset(subsample, domain=self.domain)

    def even_split(self, k=5, seed=0):
        key = random.PRNGKey(seed)
        datasets = []
        index = jnp.array(list(self.df.index))
        random.shuffle(key, index)
        df_split = jnp.array_split(index, k)
        for kth in df_split:
            df = self.df.loc[jnp.array(kth), :].copy()
            datasets.append(Dataset(df, domain=self.domain))
        return datasets


    def get_row(self, row_index):
        row_df = self.df.iloc[[row_index]]
        return Dataset(row_df, domain=self.domain)

    def get_row_dataset_list(self):
        N = len(self.df)
        res = []
        for k in range(N):
            res.append(self.get_row(k))
        return res


    @staticmethod
    def from_numpy_to_dataset(domain, X):
        df = pd.DataFrame(np.array(X), columns=domain.attrs)
        return Dataset(df, domain=domain)

    def to_numpy(self):
        array = jnp.array(self.df.values.astype(float))
        return array

    def to_numpy_np(self):
        array = np.array(self.df.values.astype(float))
        return array
    # def to_onehot(self) -> jnp.ndarray:
    #     df_data = self.df
    #     oh_encoded = []
    #     for attr, num_classes in zip(self.domain.attrs, self.domain.shape):
    #         col = df_data[attr].values
    #         if num_classes > 1:
    #             # Categorical features
    #             col_oh = jax.nn.one_hot(col.astype(int), num_classes)
    #             oh_encoded.append(col_oh)
    #         elif num_classes == 1:
    #             # Numerical features
    #             oh_encoded.append(col.astype(float).reshape((-1, 1)))
    #     data_onehot = jnp.concatenate(oh_encoded, axis=1)
    #
    #     return data_onehot.astype(float)

    # @staticmethod
    # def from_onehot_to_dataset(domain: Domain, X_oh):
    #
    #     dtypes = []
    #     start = 0
    #     output = []
    #     column_names = domain.attrs
    #     cat_cols = domain.get_categorical_cols()
    #     for col in domain.attrs:
    #         dimensions = domain.size(col)
    #         column_onehot_encoding = X_oh[:, start:start + dimensions]
    #
    #
    #         if col in cat_cols:
    #             encoder = OneHotEncoder()
    #             # encoder.categories_ = [list(range(domain.size(col)))]
    #             dummy = [[v] for v in range(domain.size(col))]
    #             encoder.fit(dummy)
    #             col_values = encoder.inverse_transform(column_onehot_encoding)
    #             dtypes.append('int64')
    #         else:
    #             col_values = column_onehot_encoding.squeeze()
    #             dtypes.append('float')
    #
    #         output.append(col_values)
    #         start += dimensions
    #
    #     data_np = np.column_stack(output)
    #
    #     data_df = pd.DataFrame(data_np, columns=column_names)
    #
    #     dtypes = pd.Series(dtypes, data_df.columns)
    #
    #     data_df = data_df.astype(dtypes)
    #
    #     data = Dataset(data_df, domain=domain)
    #
    #     return data


    # @staticmethod
    # def apply_softmax(domain: Domain, X_relaxed: jnp.ndarray) -> jnp.ndarray:
    #     # Takes as input relaxed dataset
    #     # Then outputs a dataset consistent with data schema self.domain
    #     X_softmax = []
    #     i = 0
    #     for attr, num_classes in zip(domain.attrs, domain.shape):
    #         logits = X_relaxed[:, i:i+num_classes]
    #         i += num_classes
    #
    #         if num_classes > 1:
    #             X_col = jax.nn.softmax(x=logits, axis=1)
    #             X_softmax.append(X_col)
    #         else:
    #             logits_clipped = jnp.clip(logits, a_min=0, a_max=1)
    #             X_softmax.append(logits_clipped)
    #
    #
    #     X_softmax = jnp.concatenate(X_softmax, axis=1)
    #     return X_softmax

    # @staticmethod
    # def normalize_categorical(domain: Domain, X_relaxed: jnp.ndarray) -> jnp.ndarray:
    #     # Takes as input relaxed dataset
    #     # Then outputs a dataset consistent with data schema self.domain
    #     X_softmax = []
    #     i = 0
    #     for attr, num_classes in zip(domain.attrs, domain.shape):
    #         logits = X_relaxed[:, i:i+num_classes]
    #         i += num_classes
    #
    #         if num_classes > 1:
    #             # logits = logits < 0.0001
    #             # min_r = jnp.min(jnp.array([0, jnp.min(logits)]))
    #             # min_r = jnp.min(logits)
    #             # logits = logits - min_r
    #
    #             sum_r = jnp.sum(logits, axis=1)
    #             temp = jnp.array([sum_r, 0.00001 * jnp.ones_like(sum_r)])
    #             sum_r = jnp.max(temp, axis=0)
    #             X_col = logits / sum_r.reshape(-1, 1)
    #             X_softmax.append(X_col)
    #         else:
    #             # maxval = logits.max()
    #             # minval = logits.min()
    #             # range_vals = maxval - minval
    #             # logits_norm = (logits + minval) / range_vals
    #
    #             X_softmax.append(logits)
    #
    #
    #     X_softmax = jnp.concatenate(X_softmax, axis=1)
    #     return X_softmax

    # @staticmethod
    # def get_sample_onehot(key, domain, X_relaxed: jnp.ndarray, num_samples=1) -> jnp.ndarray:
    #
    #     keys = jax.random.split(key, len(domain.attrs))
    #     X_onehot = []
    #     i = 0
    #     for attr, num_classes, subkey in zip(domain.attrs, domain.shape, keys):
    #         logits = X_relaxed[:, i:i+num_classes]
    #         i += num_classes
    #         if num_classes > 1:
    #             row_one_hot = []
    #             for _ in range(num_samples):
    #
    #                 sum_r = jnp.sum(logits, axis=1)
    #                 temp = jnp.array([sum_r, 0.00001 * jnp.ones_like(sum_r)])
    #                 sum_r = jnp.max(temp, axis=0)
    #                 logits = logits / sum_r.reshape(-1, 1)
    #
    #                 subkey, subsubkey = jax.random.split(subkey, 2)
    #                 categories = jax.random.categorical(subsubkey, jnp.log(logits), axis=1)
    #                 onehot_col = jax.nn.one_hot(categories.astype(int), num_classes)
    #                 row_one_hot.append(onehot_col)
    #             X_onehot.append(jnp.concatenate(row_one_hot, axis=0))
    #         else:
    #             row_one_hot = []
    #             # Add numerical column
    #             for _ in range(num_samples):
    #                 row_one_hot.append(logits)
    #
    #             X_onehot.append(jnp.concatenate(row_one_hot, axis=0))
    #
    #
    #     X_onehot = jnp.concatenate(X_onehot, axis=1)
    #     return X_onehot


    @staticmethod
    def apply_softmax(domain: Domain, X_relaxed: jnp.ndarray) -> jnp.ndarray:
        # Takes as input relaxed dataset
        # Then outputs a dataset consistent with data schema self.domain
        X_softmax = []
        i = 0
        for attr, num_classes in zip(domain.attrs, domain.shape):
            logits = X_relaxed[:, i:i+num_classes]
            i += num_classes

            if num_classes > 1:
                X_col = jax.nn.softmax(x=logits, axis=1)
                X_softmax.append(X_col)
            else:
                # logits_clipped = jnp.clip(logits, a_min=0, a_max=1)
                # X_softmax.append(logits_clipped)
                X_softmax.append(logits)


        X_softmax = jnp.concatenate(X_softmax, axis=1)
        return X_softmax

    @staticmethod
    def normalize_categorical(domain: Domain, X_relaxed: jnp.ndarray) -> jnp.ndarray:
        # Takes as input relaxed dataset
        # Then outputs a dataset consistent with data schema self.domain
        X_softmax = []
        i = 0
        for attr, num_classes in zip(domain.attrs, domain.shape):
            logits = X_relaxed[:, i:i+num_classes]
            i += num_classes

            if num_classes > 1:
                # logits = logits < 0.0001
                # min_r = jnp.min(jnp.array([0, jnp.min(logits)]))
                # min_r = jnp.min(logits)
                # logits = logits - min_r

                sum_r = jnp.sum(logits, axis=1)
                temp = jnp.array([sum_r, 0.00001 * jnp.ones_like(sum_r)])
                sum_r = jnp.max(temp, axis=0)
                X_col = logits / sum_r.reshape(-1, 1)
                X_softmax.append(X_col)
            else:
                # maxval = logits.max()
                # minval = logits.min()
                # range_vals = maxval - minval
                # logits_norm = (logits + minval) / range_vals

                X_softmax.append(logits)


        X_softmax = jnp.concatenate(X_softmax, axis=1)
        return X_softmax

    @staticmethod
    def get_sample_onehot(key, domain, X_relaxed: jnp.ndarray, num_samples=1) -> jnp.ndarray:

        keys = jax.random.split(key, len(domain.attrs))
        X_onehot = []
        i = 0
        for attr, num_classes, subkey in zip(domain.attrs, domain.shape, keys):
            logits = X_relaxed[:, i:i+num_classes]
            i += num_classes
            if num_classes > 1:
                row_one_hot = []
                for _ in range(num_samples):

                    sum_r = jnp.sum(logits, axis=1)
                    temp = jnp.array([sum_r, 0.00001 * jnp.ones_like(sum_r)])
                    sum_r = jnp.max(temp, axis=0)
                    logits = logits / sum_r.reshape(-1, 1)

                    subkey, subsubkey = jax.random.split(subkey, 2)
                    categories = jax.random.categorical(subsubkey, jnp.log(logits), axis=1)
                    onehot_col = jax.nn.one_hot(categories.astype(int), num_classes)
                    row_one_hot.append(onehot_col)
                X_onehot.append(jnp.concatenate(row_one_hot, axis=0))
            else:
                row_one_hot = []
                # Add numerical column
                for _ in range(num_samples):
                    row_one_hot.append(logits)

                X_onehot.append(jnp.concatenate(row_one_hot, axis=0))


        X_onehot = jnp.concatenate(X_onehot, axis=1)
        return X_onehot


    def discretize(self, num_bins=10):
        """
        Discretize real-valued columns using an equal sized binning strategy
        """
        numerical_cols = self.domain.get_numeric_cols()
        bin_edges = np.linspace(0, 1, num_bins+1)[1:]
        cols = []
        domain_shape = []
        for col, shape in zip(self.domain.attrs, self.domain.shape) :
            col_values = self.df[col].values
            if col in numerical_cols:
                discrete_col_values = np.digitize(col_values, bin_edges)
                cols.append(discrete_col_values)
                domain_shape.append(num_bins)
            else:
                cols.append(col_values)
                domain_shape.append(shape)

        discrete_domain = Domain(self.domain.attrs, domain_shape)
        df = pd.DataFrame(np.column_stack(cols), columns=self.domain.attrs)
        return Dataset(df, discrete_domain)

    def normalize_real_values(self):
        """
        Map real-valued to [0, 1]
        """
        numerical_cols = self.domain.get_numeric_cols()
        cols = []
        col_range = {}
        for col, shape in zip(self.domain.attrs, self.domain.shape):
            col_values = self.df[col].values
            if col in numerical_cols:
                min_val = col_values.min()
                max_val = col_values.max()
                normed_col_values = (col_values-min_val) / (max_val - min_val)
                cols.append(normed_col_values)
                col_range[col] = (min_val, max_val)
            else:
                cols.append(col_values)
        df = pd.DataFrame(np.column_stack(cols), columns=self.domain.attrs)
        return Dataset(df, self.domain), col_range

    def inverse_map_real_values(self, col_range: dict):
        """
        Map real-values to original range
        :param col_range: A dictionary where each entry is the range of each real-valued column
        """
        numerical_cols = self.domain.get_numeric_cols()
        cols = []
        for col, shape in zip(self.domain.attrs, self.domain.shape):
            col_values = np.array(self.df[col].values)
            if col in numerical_cols:
                min_val, max_val = col_range[col]
                orginal_range_col_values = col_values * (max_val - min_val) + min_val
                cols.append(orginal_range_col_values)
            else:
                cols.append(col_values)
        df = pd.DataFrame(np.column_stack(cols), columns=self.domain.attrs)
        return Dataset(df, self.domain)

    @staticmethod
    def to_numeric(data, numeric_features: list):
        cols = []
        domain_shape = []
        """ when called on a discretized dataset it turns discretized features into numeric features """
        for col, shape in zip(data.domain.attrs, data.domain.shape):
            col_values = data.df_real[col].values

            if col in numeric_features:
                rand_values = np.random.rand(col_values.shape[0]) / shape
                numeric_values = col_values / shape
                numeric_values = numeric_values + rand_values
                cols.append(numeric_values)
                domain_shape.append(1)
            else:
                cols.append(col_values)
                domain_shape.append(shape)
        numeric_domain = Domain(data.domain.attrs, domain_shape)
        df = pd.DataFrame(np.column_stack(cols), columns=data.domain.attrs)
        return Dataset(df, numeric_domain)

    def split(self, p, seed=0):
        np.random.seed(seed)
        msk = np.random.rand(len(self.df)) < p
        train = self.df[msk]
        test = self.df[~msk]
        return Dataset(train, self.domain), Dataset(test, self.domain)
##################################################
# Test
##################################################


if __name__ == "__main__":
    # test_onehot_encoding()
    # test_discrete()

    df = pd.read_csv('../tests/data.csv')
    config = json.load(open('../tests/domain.json'))

    print(df)
    print(config)
    domain = Domain(config)

    data = Dataset(df, domain)


    print(data.to_numpy())
    print(data.df)