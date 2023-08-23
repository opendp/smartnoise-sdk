import subprocess
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
# from snsynth.gsd import GSDSynthesizer
from snsynth.gsd.utils.dataset_jax import Dataset, Domain
from snsynth.gsd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, _get_density_fn, get_quantiles



class TestStatistics:

    def test_private_quantiles(self):

        meta_data = {'O1': {'type': 'int', 'size': 10000}}

        N = 1000
        values = np.concatenate((np.zeros(91).astype(int), np.random.randint(0, 10000, size=(9, ))))
        df = pd.DataFrame({'O1': values})
        # synth = GSDSynthesizer(10000.0, 1e-5, verbose=True)

        domain = Domain(config=meta_data)
        data = Dataset(df, domain)

        # data = synth._get_data(df, meta_data=meta_data)

        bin_edges = _get_bin_edges(domain=data.domain, tree_height=20)
        _, _, marginals_info = _get_mixed_marginal_fn(data, k=1, bin_edges=bin_edges, rho=None,
                                                      store_marginal_stats=True,
                                                      verbose=True)

        approx_quantiles, densities, threshold, threshold_density = get_quantiles(marginals_info[('O1',)], num_quantiles=20)

        print()



