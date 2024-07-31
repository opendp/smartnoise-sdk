import unittest
import numpy as np
import pandas as pd
from snsynth.gsd import GSDSynthesizer
from snsynth.gsd.utils import Dataset, Domain
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, _get_density_fn, get_quantiles


class TestFit(unittest.TestCase):


    ## Test fit
    def test_mixed_columns_fit(self):
        columns = ['R1', 'R2', 'C1', 'O1', 'C2']

        meta_data = {
            'R1': {'type': 'float', 'lower': 0, 'upper': 1},
            'R2': {'type': 'float', 'lower': 0, 'upper': 10},
            'C1': {'type': 'string'},
            'O1': {'type': 'int', 'lower': 0, 'upper': 100},
            'C2': {'type': 'string'},
        }

        data_array = [
            [0.01, 1.1, 'Cat', 0, 0],
            [0.06, 2.8, 'Cat', 4, 0],
            [0.09, 0.2, 'Dog', 4, 1],
            [0.02, 2.3, 'Cat', 0, 0],
            [0.11, 0.1, 'Dog', 9, 1],
            [0.18, 7.2, 'Bird', 4, 1],
            [0.12, 0.9, 'Fish', 8, 0],
            [0.13, 0.9, 'Fish', 8, 0],
            [0.15, 0.9, 'Cat', 4, 0],
            [0.14, 0.9, 'Fish', 8, 0],
        ]
        df = pd.DataFrame(data_array, columns=columns)
        synth = GSDSynthesizer(10000000.0, 1e-5, verbose=True)
        synth.fit(df, meta_data=meta_data, seed=0)
        print(synth.sample())
        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        assert max_error < 0.001



    def test_2d_continuous(self):

        columns = ['R1', 'R2']
        col_bounds = {'R1': {'type': 'float', 'lower':0, 'upper': 1}, 'R2': {'type': 'float', 'lower': 0, 'upper': 1}}
        N = 10
        data_array = np.column_stack([
            0.37 * np.ones(N),
            np.concatenate([0.64 * np.ones(N//2), np.zeros(N//2)]),
            ])
        df = pd.DataFrame(data_array, columns=columns)

        # Since we are passing the data bounds, we do not need to provide privacy budget for preprocessing.
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df,  meta_data=col_bounds)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        assert max_error < 0.001

    def test_3d_continuous(self):


        meta_data = {'R1': {'type': 'float', 'lower':0, 'upper': 1}, 'R2': {'type': 'float', 'lower': 0, 'upper': 1},
                     'R3': {'type': 'float', 'lower': 0, 'upper': 1}}

        N = 10
        data_array = {
            'R1': 0.37 * np.ones(N),
            'R2': np.concatenate([0.64 * np.ones(N//2), np.zeros(N//2)]),
            'R3': np.concatenate([0.08 * np.ones(N//5), 0.99 * np.ones(4*N//5)]),
        }
        df = pd.DataFrame(data_array)

        # Since we are passing the data bounds, we do not need to provide privacy budget for preprocessing.
        print('Mutate only:')
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, meta_data=meta_data)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(max_error)
        assert max_error < 0.001


    def test_2d_ordinal(self):
        rng = np.random.default_rng(0)
        columns = ['O1', 'O2']
        col_bounds = {'O1': {'type': 'int', 'lower':0, 'upper': 100}, 'O2': {'type': 'int', 'lower': 0, 'upper': 100}}
        N = 10
        data_array = np.column_stack([
            10 * np.ones(N),
            np.concatenate([rng.integers(0, 100, N//2), np.zeros(N//2)]),
            ])
        df = pd.DataFrame(data_array, columns=columns)
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, ordinal_columns=columns, meta_data=col_bounds)
        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(f'max_error = {max_error:.3f}')
        assert max_error < 0.001

    def test_categorical_ordinal(self):
        rng = np.random.default_rng(1)
        columns = ['O1', 'Cat1', 'Cat2']
        col_bounds = {'O1': {'type': 'int', 'lower':0, 'upper': 100}}
        N = 10
        data_array = {
            'O1': np.concatenate([rng.integers(1, 100, 3*N//5), np.zeros(2*N//5)]).astype(int), # Ordinal values
            'Cat1': np.concatenate([rng.integers(1, 3, N//2), np.zeros(N//2)]).astype(int).astype(str),   # Categorical values
            'Cat2': np.concatenate([rng.integers(1, 3, N//2), np.zeros(N//2)]).astype(int).astype(str),   # Categorical values
        }
        df = pd.DataFrame(data_array)
        synth = GSDSynthesizer(1000000.0, 1e-5, verbose=True)
        synth.fit(df, ordinal_columns=['O1'], categorical_columns=['Cat1', 'Cat2'], meta_data=col_bounds)

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
        synth.fit(df, continuous_columns=['Cont1', 'Cont2'], categorical_columns=['Cat1'], meta_data=col_bounds)

        max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
        print(f'max_error = {max_error:.3f}')
        assert max_error < 0.001

if __name__ == '__main__':
    unittest.main()


