import seaborn as sns
import numpy as np
from snsynth.transform.label import LabelTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t]) for t in iris.itertuples()]

labels_orig = [row[5] for row in iris]
sepal_orig = [row[1] for row in iris]

class TestLabelEncoder:
    def test_label_by_colidx(self):
        le = LabelTransformer()
        le.fit(iris, 5)
        categories = le.transform(labels_orig)
        labels_after = le.inverse_transform(categories)
        assert(all([a == b for a, b in zip(labels_orig, labels_after)]))
    def test_label_by_vals(self):
        le = LabelTransformer()
        le.fit(labels_orig)
        categories = le.transform(labels_orig)
        labels_after = le.inverse_transform(categories)
        assert(all([a == b for a, b in zip(labels_orig, labels_after)]))
    def test_label_by_vals_and_colidx(self):
        le = LabelTransformer()
        le.fit(iris, 5)
        categories = le.transform(labels_orig)
        labels_after = le.inverse_transform(categories)
        assert(all([a == b for a, b in zip(labels_orig, labels_after)]))
    def test_label_by_vals_and_colidx_with_fit_transform(self):
        le = LabelTransformer()
        categories = le.fit_transform(iris, 5)
        labels_after = le.inverse_transform(categories)
        assert(all([a == b for a, b in zip(labels_orig, labels_after)]))

