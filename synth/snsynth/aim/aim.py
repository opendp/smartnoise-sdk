import numpy as np
import pandas as pd

try:
    from mbi import FactoredInference, Dataset, Domain, GraphicalModel
except ImportError:
    print("Please install mbi with:\n   pip install git+https://github.com/ryan112358/private-pgm.git")

import argparse
import itertools
from snsynth.base import Synthesizer
from mbi import Dataset, FactoredInference, Domain
from hdmm.matrix import Identity
from snsynth.cdp2adp import cdp_rho
from snsynth.utils import exponential_mechanism, gaussian_noise, powerset

prng = np.random


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2 ** 20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class AIMSynthesizer(Synthesizer):
    """AIM: An Adaptive and Iterative Mechanism

    Based on the code available ins
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py

    #TODO: describe params
    """

    def __init__(self, epsilon=1., delta=1e-9, max_model_size=80, degree=2, num_marginals=None, max_cells: int = 10000,
                 rounds=None, verbose=False):
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.max_cells = max_cells
        self.degree = degree
        self.num_marginals = num_marginals
        self.verbose = verbose
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.synthesizer = None

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
            prng=prng,
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
        if any(c is None for c in cards):
            raise ValueError("The transformer appears to have some continuous columns. Please provide only categorical or ordinal.")

        dimensionality = np.prod(cards)
        if self.verbose:
            print(f"Fitting with {dimensionality} dimensions")

        colnames = ["col" + str(i) for i in range(self._transformer.output_width)]

        if len(cards) != len(colnames):
            raise ValueError("Cardinality and column names must be the same length.")

        domain = Domain(colnames, cards)

        data = pd.DataFrame(train_data, columns=colnames)
        data = Dataset(df=data, domain=domain)
        workload = self.get_workload(
            data, degree=self.degree, max_cells=self.max_cells, num_marginals=self.num_marginals
        )

        self.AIM(data, workload)

    def sample(self, samples=None):
        data = self.synthesizer.synthetic_data(rows=samples)
        return data

    @staticmethod
    def get_workload(data: Dataset, degree: int, max_cells: int, num_marginals: int = None):
        workload = list(itertools.combinations(data.domain, degree))
        workload = [cl for cl in workload if data.domain.size(cl) <= max_cells]
        if num_marginals is not None:
            workload = [workload[i] for i in prng.choice(len(workload), num_marginals, replace=False)]

        # workload = [(cl, 1.0) for cl in workload]
        return workload

    def _worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(sensitivity.values())  # if all weights are 0, could be a problem
        return exponential_mechanism(errors, eps, max_sensitivity)

    def AIM(self, data, workload):
        rounds = self.rounds or 16 * len(data.domain)
        # workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        print('Initial Sigma', sigma)
        rho_used = len(oneway) * 0.5 / sigma ** 2
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + gaussian_noise(sigma, x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))

        engine = FactoredInference(data.domain, iters=1000, warm_start=True)
        model = engine.estimate(measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self._worst_approximated(small_candidates, answers, model, epsilon, sigma)

            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()

            model = engine.estimate(measurements)
            w = model.project(cl).datavector()
            print('Selected', cl, 'Size', n, 'Budget Used', rho_used / self.rho)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma / 2)
                sigma /= 2
                epsilon *= 2

        engine.iters = 2500
        model = engine.estimate(measurements)

        if self.verbose:
            print("Estimating marginals")

        self.synthesizer = model

    def get_errors(self, data: Dataset, workload):
        errors = []
        for proj, wgt in workload:
            X = data.project(proj).datavector()
            Y = self.synthesizer.project(proj).datavector()
            e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
            errors.append(e)
        print('Average Error: ', np.mean(errors))
        return errors


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = '/home/stacy/GitHub/private-pgm/data/adult.csv'
    params['domain'] = '/home/stacy/GitHub/private-pgm/data/adult-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['max_model_size'] = 80
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000

    return params


def main(dataset, domain, epsilon, delta, max_model_size, degree, num_marginals, max_cells, save):

    data = Dataset.load(dataset, domain)

    synth = AIMSynthesizer(epsilon=epsilon, delta=delta, max_model_size=max_model_size, degree=degree,
                           max_cells=max_cells, num_marginals=num_marginals)

    synth.fit(data.df)
    # errors = synth.get_errors(data, workload)
    df = synth.sample(len(data.df))

    if save is not None:
        df.to_csv('aim_sample.csv', index=False)


if __name__ == "__main__":

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')
    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')
    parser.add_argument('--save', type=str, help='path to save synthetic data')

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    main(args.dataset, args.domain, args.epsilon, args.delta, args.max_model_size,
         args.degree, args.num_marginals, args.max_cells, args.save)
