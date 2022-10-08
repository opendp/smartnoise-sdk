import seaborn as sns
import numpy as np
from torch import neg
from snsynth.transform.minmax import MinMaxTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t]) for t in iris.itertuples()]

labels_orig = [row[5] for row in iris]
sepal_orig = [row[1] for row in iris]

class TestMinMax:
    def test_minmax_close_bounds(self):
        mmt = MinMaxTransformer(lower=4, upper=8, negative=False)
        sepal_encoded = mmt.transform(sepal_orig)
        assert(np.max(sepal_encoded) <= 1)
        assert(np.min(sepal_encoded) >= 0)
        sepal_decoded = mmt.inverse_transform(sepal_encoded)
        assert(all([a == b for a, b in zip(sepal_orig, sepal_decoded)]))
    def test_minmax_neg_close_bounds(self):
        mmt = MinMaxTransformer(lower=4, upper=8, negative=True)
        sepal_encoded = mmt.transform(sepal_orig)
        assert(max(sepal_encoded) <= 1)
        assert(min(sepal_encoded) >= -1)
        assert(min(sepal_encoded) < 0)
        sepal_decoded = mmt.inverse_transform(sepal_encoded)
        assert(all([a == b for a, b in zip(sepal_orig, sepal_decoded)]))
    def test_minmax_neg_approx_bounds_neg(self):
        mmt = MinMaxTransformer(negative=True, epsilon=1.0)
        sepal_encoded = mmt.fit_transform(sepal_orig)
        assert(max(sepal_encoded) <= 1)
        assert(min(sepal_encoded) >= -1)
        assert(min(sepal_encoded) < 0)
        sepal_decoded = mmt.inverse_transform(sepal_encoded)
        assert(all([a == b for a, b in zip(sepal_orig, sepal_decoded)]))
    def test_minmax_neg_approx_bounds_pos(self):
        mmt = MinMaxTransformer(epsilon=1.0, negative=False)
        sepal_encoded = mmt.fit_transform(sepal_orig)
        assert(max(sepal_encoded) <= 1.0)
        assert(min(sepal_encoded) >= 0.0)
        sepal_decoded = mmt.inverse_transform(sepal_encoded)
        assert(all([a == b for a, b in zip(sepal_orig, sepal_decoded)]))