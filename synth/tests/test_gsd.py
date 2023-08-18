import subprocess
import os

import numpy as np
import pandas as pd

from snsynth.gsd import GSDSynthesizer


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

df = pd.read_csv(csv_path, index_col=None)
df = df.drop(["income"], axis=1)
nf = df.to_numpy().astype(int)

test_data = np.array([[1,1,1],[2,2,2],[3,3,3]])

test_histogram = [[[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],
        [[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],
        [[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]]]

test_histogram_dims = (3,3,3)


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

    def test_1d(self):
        data_array, columns = get_simple_cat_data()
        df = pd.DataFrame(data_array, columns=columns)
        df = df[['C3']]

        synth = GSDSynthesizer(500000.0, 1e-5, verbose=True)
        synth.fit(df, categorical_columns=['C3'])

        true_stats = synth.stat_fn(synth.data.to_numpy())
        sync_stats = synth.stat_fn(synth.sync_data.to_numpy())
        max_error = np.abs(true_stats - sync_stats).max()
        print(f'final max error = {max_error:.5f}')

        sync_data = synth.sample()
        print(sync_data)
        print()



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

        synth.fit(data_array, preprocessor_eps=1000)
        assert synth.categorical_columns == [2]
        assert synth.ordinal_columns == [3]
        assert synth.continuous_columns == [0, 1]

        df = pd.DataFrame(data_array, columns=cont_colums + cat_colums + ord_colums)
        synth.fit(df, preprocessor_eps=1000)

        assert synth.categorical_columns == ['C1']
        assert synth.ordinal_columns == ['O1']
        assert synth.continuous_columns == ['R1', 'R2']

        # Pass data bounds directly
    def test_mixed_columns_w_known_bounds(self):
        ## NOTE: Not passing.

        cont_colums = ['R1', 'R2']
        cat_colums = ['C1', 'C2']
        ord_colums = ['O1']

        columns = ['R1', 'R2', 'C1', 'O1', 'C2']

        col_bounds = {
            'R1': {'lower':0, 'upper': 1},
            'R2': {'lower': 0, 'upper': 10},
            'O1': {'lower': 0, 'upper': 100},
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
        df = df[cont_colums + cat_colums]

        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        # Since we are passing the data bounds, we do not need to provide privacy budget for preprocessing.
        synth.fit(df,
                  # genetic_operators=['mutate'],
                  tree_height=64,
                  categorical_columns=cat_colums,
                  # ordinal_columns=ord_colums,
                  continuous_columns=cont_colums,
                  data_bounds=col_bounds)

        sync_df = synth.sample()
        print(sync_df)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(f'max_error = {max_error:.3f}')
        assert max_error < 0.001
        assert synth.categorical_columns == ['C1', 'C2']
        assert synth.ordinal_columns == ['O1']
        assert synth.continuous_columns == ['R1', 'R2']

    def test_2d_continuous(self):

        columns = ['R1', 'R2']
        col_bounds = {'R1': {'lower':0, 'upper': 1}, 'R2': {'lower': 0, 'upper': 1}}
        N = 10
        data_array = np.column_stack([
            0.37 * np.ones(N),
            np.concatenate([0.64 * np.ones(N//2), np.zeros(N//2)]),
        ])
        df = pd.DataFrame(data_array, columns=columns)

        # Since we are passing the data bounds, we do not need to provide privacy budget for preprocessing.
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, continuous_columns=columns, data_bounds=col_bounds)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        assert max_error < 0.001

    def test_3d_continuous(self):

        columns = ['R1', 'R2', 'R3']

        col_bounds = {'R1': {'lower':0, 'upper': 1}, 'R2': {'lower': 0, 'upper': 1}, 'R3': {'lower': 0, 'upper': 1}}

        N = 10
        data_array = {
            'R1':0.37 * np.ones(N),
            'R2':np.concatenate([0.64 * np.ones(N//2), np.zeros(N//2)]),
            'R3':np.concatenate([0.08 * np.ones(N//5), 0.99 * np.ones(4*N//5)]),
        }
        df = pd.DataFrame(data_array, columns=columns)

        # Since we are passing the data bounds, we do not need to provide privacy budget for preprocessing.
        print('Mutate only:')
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, continuous_columns=columns, data_bounds=col_bounds)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(max_error)
        assert max_error < 0.001

    def test_simple_categorical(self):
        data_array, columns = get_simple_cat_data()
        df = pd.DataFrame(data_array, columns=columns)

        synth = GSDSynthesizer(500000.0, 1e-5, verbose=True)
        synth.fit(df, categorical_columns=columns)

        true_stats = synth.stat_fn(synth.data.to_numpy())
        sync_stats = synth.stat_fn(synth.sync_data.to_numpy())
        max_error = np.abs(true_stats - sync_stats).max()
        assert max_error < 0.001



    def test_2d_ordinal(self):
        rng = np.random.default_rng(0)
        columns = ['O1', 'O2']
        col_bounds = {'O1': {'lower':0, 'upper': 100}, 'O2': {'lower': 0, 'upper': 100}}
        N = 10
        data_array = np.column_stack([
            10 * np.ones(N),
            np.concatenate([rng.integers(0, 100, N//2), np.zeros(N//2)]),
            ])
        df = pd.DataFrame(data_array, columns=columns)
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, ordinal_columns=columns, data_bounds=col_bounds)
        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(f'max_error = {max_error:.3f}')
        assert max_error < 0.001

    def test_categorical_ordinal(self):
        rng = np.random.default_rng(1)
        columns = ['O1', 'Cat1', 'Cat2']
        col_bounds = {'O1': {'lower':0, 'upper': 100}}
        N = 10
        data_array = np.column_stack([
            np.concatenate([rng.integers(1, 100, 3*N//5), np.zeros(2*N//5)]).astype(int), # Ordinal values
            np.concatenate([rng.integers(1, 3, N//2), np.zeros(N//2)]).astype(int).astype(str),   # Categorical values
            np.concatenate([rng.integers(1, 3, N//2), np.zeros(N//2)]).astype(int).astype(str),   # Categorical values
            ])
        df = pd.DataFrame(data_array, columns=columns)
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, ordinal_columns=['O1'], categorical_columns=['Cat1', 'Cat2'], data_bounds=col_bounds)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(f'max_error = {max_error:.3f}')
        assert max_error < 0.001



    def test_categorical_continuous(self):
        rng = np.random.default_rng(1)
        columns = ['Cont1', 'Cont2', 'Cat1']
        col_bounds = {'Cont1': {'lower':0, 'upper': 1}, 'Cont2': {'lower':0, 'upper': 10}}
        N = 10
        data_array = {
            'Cont1':np.concatenate([rng.normal(0.5, 0.1, 3*N//5), 0.2 * np.ones(2*N//5)]),
            'Cont2':np.concatenate([rng.normal(5, 2, 3*N//5), 2 * np.ones(2*N//5)]),
            'Cat1':np.concatenate([rng.integers(1, 3, N//2), np.zeros(N//2)]).astype(int).astype(str),   # Categorical values
        }
        df = pd.DataFrame(data_array, columns=columns)
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, continuous_columns=['Cont1', 'Cont2'], categorical_columns=['Cat1'], data_bounds=col_bounds)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(f'max_error = {max_error:.3f}')
        assert max_error < 0.001
    def test_null(self):
        pass
