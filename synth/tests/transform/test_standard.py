import seaborn as sns
import numpy as np
from snsynth.transform import StandardScaler
from snsynth.transform.minmax import MinMaxTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t]) for t in iris.itertuples()]

labels_orig = [row[5] for row in iris]
sepal_orig = [row[1] for row in iris]

class TestStandardScaler:
    def test_ss_close_bounds(self):
        mmt = StandardScaler(lower=4, upper=8, epsilon=10.0)
        mmt.fit(sepal_orig)
        sepal_encoded = mmt.transform(sepal_orig)
        sepal_decoded = mmt.inverse_transform(sepal_encoded)
        assert(all([np.isclose(a,b) for a, b in zip(sepal_orig, sepal_decoded)]))
    def test_ss_approx_bounds_pos(self):
        mmt = StandardScaler(lower=4, upper=8, epsilon=10.0)
        sepal_encoded = mmt.fit_transform(sepal_orig)
        sepal_decoded = mmt.inverse_transform(sepal_encoded)
        assert(all([np.isclose(a,b) for a, b in zip(sepal_orig, sepal_decoded)]))