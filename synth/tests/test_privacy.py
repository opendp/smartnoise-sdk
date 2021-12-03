from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import DPCTGAN, PATECTGAN
import numpy as np
import pandas as pd
from tqdm import trange
from scipy.stats import ttest_ind
import os

import warnings
warnings.filterwarnings("ignore")

# from tests.test_privacy import TestSynthPrivacy
# tp = TestSynthPrivacy()
# tp.evaluate_dpctgan_privacy_categorical()


trials = 104
keep = 100
size = 5000
eps = 1.0 / size

X, Y = "X", "Y"
categorical_a = pd.DataFrame(columns=["A"], data=np.array([[Y] + [X] * size]).T)
categorical_b = pd.DataFrame(columns=["B"], data=np.array([[X] + [Y] * size]).T)

def check_stats_for_epsilon(name, a, b, epsilon, keep):
    name = f"{name}_{epsilon}"
    if not os.path.exists(name):
        os.mkdir(name)
    a_out = open(os.path.join(name, "a.txt"), "w")
    a_out.writelines(str(v) + "\n" for v in a)

    b_out = open(os.path.join(name, "b.txt"), "w")
    b_out.writelines(str(v) + "\n" for v in b)

    a_out.close()
    b_out.close()

    assert((len(a) - keep) % 2 == 0)
    trim = (len(a) - keep) // 2
    if trim > 0:
        a = sorted(a)[trim:-trim]
        b = sorted(b)[trim:-trim]

    lower = np.min([np.min(a), np.min(b)])
    upper = np.max([np.max(a), np.max(b)])

    hist_a, _ = np.histogram(a, bins=10, range=(lower, upper))
    hist_b, _ = np.histogram(b, bins=10, range=(lower, upper))

    a_b = hist_a * np.exp(epsilon) - hist_b
    a_b_over = a_b < 0.0
    b_over = sum([v for v, over in list(zip(hist_b, a_b_over)) if over])

    b_a = hist_b * np.exp(epsilon) - hist_a
    b_a_over = b_a < 0.0
    a_over = sum([v for v, over in list(zip(hist_a, b_a_over)) if over])

    over = (a_over + b_over) / (len(a) * 2)
    over_out = open(os.path.join(name, "over.txt"), "w")
    over_out.write(str(over))

    print(f"{name} over: {over}")

    return over

class TestSynthPrivacy:
    def evaluate_dpctgan_privacy_categorical_group(self):
        a = []
        b = []
        for _ in trange(trials):
            gan = DPCTGAN(epsilon=eps, verbose=True)
            gan.train(categorical_a, categorical_columns=["A"])
            sample = gan.generate(size)
            a.append(sum([s == 'X' for s in sample['A']]))
            gan = DPCTGAN(epsilon=eps, verbose=True)
            gan.train(categorical_b, categorical_columns=["B"])
            sample = gan.generate(size)
            b.append(sum([s == 'X' for s in sample['B']]))
        check_stats_for_epsilon("dpctgan_group", a, b, eps, keep)
        assert(ttest_ind(a, b, equal_var=False).pvalue > 0.4)

    def evaluate_patectgan_privacy_categorical(self):
        a = []
        b = []
        eps = 1.0
        categorical_a = pd.DataFrame(columns=["A"], data=np.array([[Y] + [X] * size]).T)
        categorical_b = pd.DataFrame(columns=["B"], data=np.array([[Y] + [Y] + [X] * size]).T)
        for _ in trange(trials):
            gan = PATECTGAN(epsilon=eps, verbose=False)
            gan.train(categorical_a, categorical_columns=["A"])
            sample = gan.generate(size)
            a.append(sum([s == 'X' for s in sample['A']]))
            gan = PATECTGAN(epsilon=eps, verbose=False)
            gan.train(categorical_b, categorical_columns=["B"])
            sample = gan.generate(size)
            b.append(sum([s == 'X' for s in sample['B']]))
        check_stats_for_epsilon("patectgan", a, b, eps, keep)
    def evaluate_dpctgan_privacy_categorical(self):
        a = []
        b = []
        eps = 1.0
        categorical_a = pd.DataFrame(columns=["A"], data=np.array([[Y] + [X] * size]).T)
        categorical_b = pd.DataFrame(columns=["B"], data=np.array([[Y] + [Y] + [X] * size]).T)
        for _ in trange(trials):
            gan = DPCTGAN(epsilon=eps, verbose=False)
            gan.train(categorical_a, categorical_columns=["A"])
            sample = gan.generate(size)
            a.append(sum([s == 'X' for s in sample['A']]))
            gan = DPCTGAN(epsilon=eps, verbose=False)
            gan.train(categorical_b, categorical_columns=["B"])
            sample = gan.generate(size)
            b.append(sum([s == 'X' for s in sample['B']]))
        check_stats_for_epsilon("dpctgan", a, b, eps, keep)
