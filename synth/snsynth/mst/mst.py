import numpy as np
import pandas as pd

try:
    from mbi import FactoredInference, Dataset, Domain
except ImportError:
    print("Please install mbi with:\n   pip install git+https://github.com/ryan112358/private-pgm.git@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca")
    raise ImportError
from scipy import sparse
from disjoint_set import DisjointSet
import networkx as nx
import itertools
from scipy.special import logsumexp
from snsynth.base import Synthesizer
from snsynth.utils import cdp_rho, gaussian_noise

"""
Wrapper for MST synthesizer from Private PGM:
https://github.com/ryan112358/private-pgm/tree/e9ea5fcac62e2c5b92ae97f7afe2648c04432564

This is a generalization of the winning mechanism from the
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.
"""


class MSTSynthesizer(Synthesizer):
    """Maximum Spanning Tree synthesizer, uses Private PGM.

    :param epsilon: privacy budget for the synthesizer
    :type epsilon: float
    :param delta: privacy parameter.  Should be small, in the range of 1/(n * sqrt(n))
    :type delta: float
    :param verbose: print diagnostic information during processing
    :type verbose: bool

    Reuses code and modifies it lightly from
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py. Awesome work McKenna et. al!
    """

    def __init__(self,
                 epsilon=0.1,
                 delta=1e-9,
                 *ignore,
                 verbose=False
                 ):
        if isinstance(epsilon, int):
            epsilon = float(epsilon)
        self.epsilon = epsilon
        self.delta = delta
        self.verbose = verbose

        self.synthesizer = None
        self.num_rows = None

    def fit(
        self, 
        data, 
        *ignore,
        transformer=None,
        categorical_columns=[], 
        ordinal_columns=[],
        continuous_columns=[],
        preprocessor_eps=0.0,
        nullable=False,
        ):

        train_data = self._get_train_data(
            data,
            style='cube',
            transformer=transformer,
            categorical_columns=categorical_columns, 
            ordinal_columns=ordinal_columns, 
            continuous_columns=continuous_columns, 
            nullable=nullable,
            preprocessor_eps=preprocessor_eps
        )

        if self._transformer is None:
            raise ValueError("We weren't able to fit a transformer to the data. Please check your data and try again.")

        cards = self._transformer.cardinality
        if any (c is None for c in cards):
            raise ValueError("The transformer appears to have some continuous columns. Please provide only categorical or ordinal.")

        dimensionality = np.prod(cards)
        if self.verbose:
            print(f"Fitting with {dimensionality} dimensions")

        colnames = ["col" + str(i) for i in range(self._transformer.output_width)]
        cards = self._transformer.cardinality

        if len(cards) != len(colnames):
            raise ValueError("Cardinality and column names must be the same length.")

        domain = Domain(colnames, cards)
        self.num_rows = len(data)

        data = pd.DataFrame(train_data, columns=colnames)
        data = Dataset(df=data, domain=domain)

        self.MST(data, self.epsilon, self.delta)

    def sample(self, samples=None):
        if samples is None:
            samples = self.num_rows
        data = self.synthesizer.synthetic_data(rows=samples)
        decompressed = self.undo_compress_fn(data)
        data_iter = [tuple([c for c in t[1:]]) for t in decompressed.df.itertuples()]
        return self._transformer.inverse_transform(data_iter)

    def MST(self, data, epsilon, delta):
        rho = cdp_rho(epsilon, delta)
        sigma = np.sqrt(3/(2*rho))
        cliques = [(col,) for col in data.domain]
        if self.verbose:
            print("Getting cliques")
        log1 = self.measure(data, cliques, sigma)
        data, log1, undo_compress_fn = self.compress_domain(data, log1)

        # Here's the decompress function
        self.undo_compress_fn = undo_compress_fn

        cliques = self.select(data, rho/3.0, log1)
        log2 = self.measure(data, cliques, sigma)
        engine = FactoredInference(data.domain, iters=5000)
        if self.verbose:
            print("Estimating marginals")
        est = engine.estimate(log1+log2)

        # Here's the synthesizer

        self.synthesizer = est

    def measure(self, data, cliques, sigma, weights=None):
        if weights is None:
            weights = np.ones(len(cliques))
        weights = np.array(weights) / np.linalg.norm(weights)
        measurements = []
        for proj, wgt in zip(cliques, weights):
            x = data.project(proj).datavector()
            y = x + gaussian_noise(sigma/wgt, x.size)
            Q = sparse.eye(x.size)
            measurements.append((Q, y, sigma/wgt, proj))
        return measurements

    def compress_domain(self, data, measurements):
        supports = {}
        new_measurements = []
        for Q, y, sigma, proj in measurements:
            col = proj[0]
            sup = y >= 3*sigma
            supports[col] = sup
            if supports[col].sum() == y.size:
                new_measurements.append((Q, y, sigma, proj))
            else:  # need to re-express measurement over the new domain
                y2 = np.append(y[sup], y[~sup].sum())
                I2 = np.ones(y2.size)
                I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
                y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
                I2 = sparse.diags(I2)
                new_measurements.append((I2, y2, sigma, proj))

        undo_compress_fn = lambda data: self.reverse_data(data, supports)  # noqa: E731

        return self.transform_data(data, supports), new_measurements, undo_compress_fn

    def exponential_mechanism(self, q, eps, sensitivity, prng=np.random, monotonic=False):
        coef = 1.0 if monotonic else 0.5
        scores = coef*eps/sensitivity*q
        probas = np.exp(scores - logsumexp(scores))
        return prng.choice(q.size, p=probas)

    def select(self, data, rho, measurement_log, cliques=[]):
        engine = FactoredInference(data.domain, iters=1000)
        est = engine.estimate(measurement_log)

        weights = {}
        candidates = list(itertools.combinations(data.domain.attrs, 2))
        for a, b in candidates:
            xhat = est.project([a, b]).datavector()
            x = data.project([a, b]).datavector()
            weights[a, b] = np.linalg.norm(x - xhat, 1)

        T = nx.Graph()
        T.add_nodes_from(data.domain.attrs)
        ds = DisjointSet()

        for e in cliques:
            T.add_edge(*e)
            ds.union(*e)

        r = len(list(nx.connected_components(T)))
        epsilon = np.sqrt(8*rho/(r-1))
        for i in range(r-1):
            candidates = [e for e in candidates if not ds.connected(*e)]
            wgts = np.array([weights[e] for e in candidates])
            idx = self.exponential_mechanism(wgts, epsilon, sensitivity=1.0)
            e = candidates[idx]
            T.add_edge(*e)
            ds.union(*e)

        return list(T.edges)

    def transform_data(self, data, supports):
        df = data.df.copy()
        newdom = {}
        for col in data.domain:
            support = supports[col]
            size = support.sum()
            newdom[col] = int(size)
            if size < support.size:
                newdom[col] += 1
            mapping = {}
            idx = 0
            for i in range(support.size):
                mapping[i] = size
                if support[i]:
                    mapping[i] = idx
                    idx += 1
            assert idx == size
            df[col] = df[col].map(mapping)
        newdom = Domain.fromdict(newdom)
        return Dataset(df, newdom)

    def reverse_data(self, data, supports):
        df = data.df.copy()
        newdom = {}
        for col in data.domain:
            support = supports[col]
            mx = support.sum()
            newdom[col] = int(support.size)
            idx, extra = np.where(support)[0], np.where(~support)[0]
            mask = df[col] == mx
            if extra.size == 0:
                pass
            else:
                df.loc[mask, col] = np.random.choice(extra, mask.sum())
            df.loc[~mask, col] = idx[df.loc[~mask, col]]
        newdom = Domain.fromdict(newdom)
        return Dataset(df, newdom)
