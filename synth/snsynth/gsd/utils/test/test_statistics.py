import subprocess
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
# from snsynth.gsd import GSDSynthesizer
from snsynth.gsd.utils.dataset_jax import Dataset, Domain
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, _get_density_fn, get_quantiles

from snsynth.utils import cdp_rho

class TestStatistics:

    def test_quantiles(self):

        meta_data = {'O1': {'type': 'int', 'size': 10000}}

        values = np.concatenate((np.zeros(910).astype(int), np.random.randint(0, 10000, size=(90, ))))
        df = pd.DataFrame({'O1': values})
        # synth = GSDSynthesizer(10000.0, 1e-5, verbose=True)

        domain = Domain(config=meta_data)
        data = Dataset(df, domain)

        # data = synth._get_data(df, meta_data=meta_data)


        bin_edges = _get_bin_edges(domain=data.domain, tree_height=20)
        _, _, marginals_info = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges, rho=None,
                                                      store_marginal_stats=True,
                                                      verbose=True)

        approx_quantiles, densities, threshold, threshold_density = get_quantiles(marginals_info[('O1',)], num_quantiles=100)


        # assert densities == [0, 0.91, 0.96]
        print(approx_quantiles)
        print(densities)
        print()




    def test_private_quantiles(self):

        meta_data = {'O1': {'type': 'int', 'size': 10000}}

        values = np.concatenate((np.zeros(910).astype(int), np.random.randint(0, 10000, size=(90, ))))
        df = pd.DataFrame({'O1': values})
        # synth = GSDSynthesizer(10000.0, 1e-5, verbose=True)

        domain = Domain(config=meta_data)
        data = Dataset(df, domain)

        # data = synth._get_data(df, meta_data=meta_data)


        bin_edges = _get_bin_edges(domain=data.domain, tree_height=14)

        rho = cdp_rho(1.0, 1e-9)
        print(f'\nrho = {rho}')
        _, _, marginals_info = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges, rho=rho,
                                                      store_marginal_stats=True,
                                                      verbose=True)

        print('=========')
        print(marginals_info['stat'][0])
        print('=========')
        print(marginals_info['stat'][1])
        print('=========')
        print('=========')

        approx_quantiles, densities, threshold, threshold_density = get_quantiles(marginals_info[('O1',)], num_quantiles=100)


        # assert densities == [0, 0.91, 0.96]
        print(f'Num quantiles = {len(approx_quantiles)}')
        print(approx_quantiles)
        print(densities)
        print(np.diff(densities))
