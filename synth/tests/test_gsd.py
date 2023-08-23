import subprocess
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
from snsynth.gsd import GSDSynthesizer
from snsynth.gsd.utils import Dataset, Domain
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, _get_density_fn, get_quantiles

"""
TODO:
Test categorical crossover.
Implement  Quantile search
Estimate quantiles using post processing. 
Implement swap genetic operator
"""
git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
meta_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))
# csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

# df = pd.read_csv(csv_path, index_col=None)
# df = df.drop(["income"], axis=1)
# nf = df.to_numpy().astype(int)
#
# test_data = np.array([[1,1,1],[2,2,2],[3,3,3]])
#
# test_histogram = [[[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]],
#         [[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]],
#         [[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]]]
#
# test_histogram_dims = (3,3,3)


def get_simple_mixed_data():
    columns = ['R1', 'R2', 'C1', 'O1', 'C2']

    data_array = [
        [0.01, 1.1, 'Cat',  0, 0],
        [0.06, 2.8, 'Cat',  4, 0],
        [0.09, 0.2, 'Dog',  4, 1],
        [0.02, 2.3, 'Cat',  0, 0],
        [0.11, 0.1, 'Dog',  9, 0],
        [0.18, 7.2, 'Bird', 4, 1],
        [0.10, 0.9, 'Fish', 4, 0],
    ]

    return data_array, columns


def get_simple_cat_data():
    columns = ['C1', 'C2', 'C3', 'C4']

    data_array = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 1],
        [0, 0, 2, 2],
    ]

    return data_array, columns
class TestGSD:

    def test_infer_columns(self):

        cont_colums = ['R1', 'R2']
        cat_colums = ['C1']
        ord_colums = ['O1']

        data_array = [
            [0.01, 0.4, 'Cat', 0],
            [0.06, 0.9, 'Cat', 4],
            [0.09, 0.3, 'Dog', 4],
            [0.02, 0.3, 'Cat', 0],
            [0.11, 0.1, 'Dog', 9],
            [0.18, 0.2, 'Bird', 4],
            [0.10, 0.8, 'Fish', 4],
        ]
        synth = GSDSynthesizer(3000.0, 1e-5)

        # synth.fit(data_array, preprocessor_eps=1000)
        synth._get_data(data_array, preprocessor_eps=1000)
        assert synth.categorical_columns == [2]
        assert synth.ordinal_columns == [3]
        assert synth.continuous_columns == [0, 1]

        df = pd.DataFrame(data_array, columns=cont_colums + cat_colums + ord_colums)
        synth._get_data(df, preprocessor_eps=1000)

        assert synth.categorical_columns == ['C1']
        assert synth.ordinal_columns == ['O1']
        assert synth.continuous_columns == ['R1', 'R2']

        # Pass data bounds directly
    def test_mixed_columns_w_known_bounds(self):
        columns = ['R1', 'R2', 'C1', 'O1', 'C2']

        meta_data = {
            'C1': {'type': 'string'},
            'C2': {'type': 'string'},
            'R1': {'type': 'float', 'lower':0, 'upper': 1},
            'R2': {'type': 'float', 'lower': 0, 'upper': 10},
            'O1': {'type': 'int', 'lower': 0, 'upper': 100},
        }

        data_array = [
            [0.01, 1.1, 'Cat',  0, 0],
            [0.06, 2.8, 'Cat',  4, 0],
            [0.09, 0.2, 'Dog',  4, 1],
            [0.02, 2.3, 'Cat',  0, 0],
            [0.11, 0.1, 'Dog',  9, 1],
            [0.18, 7.2, 'Bird', 4, 1],
            [0.12, 0.9, 'Fish', 8, 0],
            [0.13, 0.9, 'Fish', 8, 0],
            [0.15, 0.9, 'Cat',  4, 0],
            [0.14, 0.9, 'Fish', 8, 0],
        ]
        df = pd.DataFrame(data_array, columns=columns)
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)

        assert synth.categorical_columns == ['C1', 'C2']
        assert synth.ordinal_columns == ['O1']
        assert synth.continuous_columns == ['R1', 'R2']

        assert  data.domain.get_categorical_cols()== ['C1', 'C2']
        assert  data.domain.get_ordinal_cols()== ['O1']
        assert  data.domain.get_continuous_cols()== ['R1', 'R2']



    ## Test statistics function

    def test_stats_1d_real(self):

        meta_data = {'R1': {'type': 'float', 'lower':0, 'upper': 1}}

        N = 1000
        df = pd.DataFrame({'R1':0.37 * np.ones(N)})
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error < 0.0001

        df = pd.DataFrame({'R1': np.ones(N)})
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error < 0.0001


        df = pd.DataFrame({'R1': np.zeros(N)})
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error < 0.0001



    def test_stats_1d_ordinal(self):

        meta_data = {'R1': {'type': 'int', 'lower':0, 'upper': 100}}

        N = 1000
        df = pd.DataFrame({'R1':37 * np.ones(N)})
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error < 0.0001

        df = pd.DataFrame({'R1': 100 * np.ones(N)})
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error < 0.0001


        df = pd.DataFrame({'R1': np.zeros(N)})
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error < 0.0001


    def test_stats_2d_real(self):

        meta_data = {'R1': {'type': 'float', 'lower':0, 'upper': 1}, 'R2': {'type': 'float', 'lower':0, 'upper': 1}}

        N = 1000
        df = pd.DataFrame({'R1': 0.37 * np.ones(N), 'R2': 0 * np.ones(N)})
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error < 0.0001

    def test_stats(self):

        meta_data = {'R1': {'lower':0, 'upper': 1}, 'R2': {'lower': 0, 'upper': 1}, 'R3': {'lower': 0, 'upper': 1}}

        N = 1000
        data_array = {
            'R1':0.37 * np.ones(N),
            'R2':np.concatenate([0.64 * np.ones(N//2), np.zeros(N//2)]),
            'R3':np.concatenate([0.08 * np.ones(N//5), 0.99 * np.ones(4*N//5)]),
        }
        df = pd.DataFrame(data_array)
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        data = synth._get_data(df, meta_data=meta_data)
        priv_stat_ans, stat_fn = synth.get_statistics(data)
        max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
        assert max_error<0.0001


    def test_stats_1d(self):
        N = 1000

        # Data 1
        df = pd.DataFrame({'R1': 0.37 * np.ones(N)})
        data1 = Dataset(df, Domain({'R1': {'type': 'float', 'size': 1}}))

        # Data 2
        df = pd.DataFrame({'R1': np.ones(N)})
        data2 = Dataset(df, Domain(config={'R1': {'type': 'float', 'size': 1}}))

        # Data 3
        df = pd.DataFrame({'R1': np.zeros(N)})
        data3 = Dataset(df, Domain(config={'R1': {'type': 'float', 'size': 1}}))

        # Data 4: ordinals
        df = pd.DataFrame({'R1': np.zeros(N).astype(int)})
        data4 = Dataset(df, Domain(config={'R1': {'type': 'int', 'size': 1000}}))

        # Data 5: ordinals
        df = pd.DataFrame({'O1': 1000 * np.ones(N).astype(int)})
        data5 = Dataset(df, Domain(config={'O1': {'type': 'int', 'size': 1000}}))

        for i, data in enumerate([data1, data2, data3, data4, data5]):
            print(f'Test data{i}')
            # numeric = data.domain.get_ordinal_cols() + data.domain.get_continuous_cols()
            bin_edges = _get_bin_edges(domain=data.domain, tree_height=10)
            priv_stat_ans, stat_fn = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges,
                                                            rho=None,
                                                            verbose=True)
            priv_stat_ans = jnp.array(priv_stat_ans)
            max_error = np.abs(priv_stat_ans - stat_fn(data.to_numpy())).max()
            assert max_error < 0.0001