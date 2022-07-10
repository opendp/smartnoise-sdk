import seaborn as sns
import numpy as np

from snsynth.transform.label import LabelTransformer
from snsynth.transform.onehot import OneHotEncoder
from snsynth.transform.chain import ChainTransformer
from snsynth.transform.table import TableTransformer
from snsynth.transform.minmax import MinMaxTransformer
from snsynth.transform.log import LogTransformer
from snsynth.transform.bin import BinTransformer
from snsynth.transform.identity import IdentityTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t[1:]]) for t in iris.itertuples()]

labels_orig = [row[4] for row in iris]
sepal_orig = [row[0] for row in iris]

sepal_large = [np.exp(v) for v in sepal_orig]

class TestTableTransform:
    def test_needs_fit(self):
        tt = TableTransformer([
            MinMaxTransformer(epsilon=1.0),
            LogTransformer(),
            ChainTransformer([
                BinTransformer(epsilon=1.0),
                LabelTransformer(),
                OneHotEncoder()
            ]),
            LogTransformer(),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
        ])
        assert(not tt.fit_complete)
        tt.fit(iter(iris))
        iris_encoded = tt.transform(iris)
        iris_decoded = tt.inverse_transform(iris_encoded)
        for row_num in np.arange(5) * 2:
            row = iris[row_num]
            for idx, v in enumerate(row):
                if idx in [0, 1, 3]:
                    # these should be very close
                    assert(np.isclose(v, iris_decoded[row_num][idx]))
                elif idx == 2:
                    # this was bin transformed, so could be off a bit
                    assert(np.abs(iris_decoded[row_num][idx] - v) < 2.0)
                else:
                    # the rest are exact
                    assert(iris_decoded[row_num][idx] == v)
    def test_iterator(self):
        tt = TableTransformer([
            MinMaxTransformer(epsilon=1.0),
            LogTransformer(),
            LogTransformer(),
            LogTransformer(),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
        ])
        assert(not tt.fit_complete)
        iris_encoded = tt.fit_transform(iter(iris))
        assert(len(iris_encoded) == 0) # this is zero because fit consumes the iterator
        iris_encoded = tt.transform(iris)
        assert(len(iris_encoded) == len(iris))
    def test_no_fit_needed(self):
        tt = TableTransformer([
            MinMaxTransformer(lower=4, upper=8),
            LogTransformer(),
            LogTransformer(),
            LogTransformer(),
            ChainTransformer([IdentityTransformer(), IdentityTransformer()]),
        ])
        assert(tt.fit_complete)
        iris_encoded = tt.transform(iris)
        iris_decoded = tt.inverse_transform(iris_encoded)
        for row_num in np.arange(5) * 2:
            row = iris[row_num]
            for idx, v in enumerate(row):
                if isinstance(v, float):
                    assert(np.isclose(v, iris_decoded[row_num][idx]))
                else:
                    assert(iris_decoded[row_num][idx] == v)
