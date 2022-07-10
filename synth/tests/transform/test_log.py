import seaborn as sns
import numpy as np
from snsynth.transform.log import LogTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t]) for t in iris.itertuples()]

sepal_orig = [np.exp(row[1] * 10) for row in iris]

class TestLog:
    def test_log_by_vals(self):
        lt = LogTransformer()
        sepal_encoded = lt.fit_transform(sepal_orig)
        sepal_decoded = lt.inverse_transform(sepal_encoded)
        assert(all([a == b for a, b in zip(sepal_orig, sepal_decoded)]))
