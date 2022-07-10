import seaborn as sns
import numpy as np
from snsynth.transform.label import LabelTransformer
from snsynth.transform.onehot import OneHotEncoder
from snsynth.transform.chain import ChainTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t]) for t in iris.itertuples()]

labels_orig = [row[5] for row in iris]
sepal_orig = [row[1] for row in iris]

class TestChain:
    def test_chain_by_colidx(self):
        ct = ChainTransformer([LabelTransformer(), OneHotEncoder()])
        labels_encoded = ct.fit_transform(iris, 5)
        assert(ct.output_width == 3)
        labels_decoded = ct.inverse_transform(labels_encoded)
        assert(all([a == b for a, b in zip(labels_orig, labels_decoded)]))

