import seaborn as sns
import numpy as np
from snsynth.transform.bin import BinTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t]) for t in iris.itertuples()]

labels_orig = [row[5] for row in iris]
sepal_orig = [row[1] for row in iris]

class TestBin:
    def test_test_bin_approx(self):
        bt = BinTransformer(bins=11, epsilon=0.1)
        bt.fit(sepal_orig)
        sepal_encoded = bt.transform(sepal_orig)
        assert(np.max(sepal_encoded) == 10)  # bins are numbered 0-10
        sepal_decoded = bt.inverse_transform(sepal_encoded)
        assert(np.mean([np.abs(a - b) for a, b in zip(sepal_orig, sepal_decoded)]) < 1.0)
    def test_test_bin_bounds(self):
        bt = BinTransformer(bins=12, lower=4, upper=8)
        sepal_encoded = bt.transform(sepal_orig)
        assert(np.max(sepal_encoded) == 11)  # bins are numbered 0-11
        sepal_decoded = bt.inverse_transform(sepal_encoded)
        assert(np.mean([np.abs(a - b) for a, b in zip(sepal_orig, sepal_decoded)]) < 1.0)
        # make sure fit doesn't change anything
        bt.fit(sepal_orig)
        sepal_encoded = bt.transform(sepal_orig)
        assert(np.max(sepal_encoded) == 11)  # bins are numbered 0-11
        sepal_decoded = bt.inverse_transform(sepal_encoded)
        assert(np.mean([np.abs(a - b) for a, b in zip(sepal_orig, sepal_decoded)]) < 1.0)
