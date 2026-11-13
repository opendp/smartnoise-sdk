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


class TestLabelPublicCategories:
    """A public list of categories: fitting must not consult the data."""
    public = ['setosa', 'versicolor', 'virginica', 'never_observed']

    def test_round_trip_with_public_categories(self):
        le = LabelTransformer(nullable=False, categories=self.public)
        le.fit(iris, 5)
        categories = le.transform(labels_orig)
        labels_after = le.inverse_transform(categories)
        assert(all([a == b for a, b in zip(labels_orig, labels_after)]))

    def test_categories_do_not_depend_on_the_data(self):
        a = LabelTransformer(nullable=False, categories=self.public)
        a.fit(labels_orig)
        b = LabelTransformer(nullable=False, categories=self.public)
        b.fit(['a value held by one row', 'setosa'])
        assert(a.categories == b.categories == self.public)
        assert(a.labels == b.labels)
        assert(a.cardinality == [len(self.public)])

    def test_order_is_the_callers(self):
        order = ['virginica', 'setosa', 'versicolor']
        le = LabelTransformer(nullable=False, categories=order)
        le.fit(labels_orig)
        assert(le.categories == order)
        assert(le.transform(['virginica', 'versicolor']) == [0, 2])

    def test_unknown_value_becomes_null_when_nullable(self):
        le = LabelTransformer(nullable=True, categories=self.public)
        le.fit(labels_orig)
        encoded = le.transform(['setosa', 'a value held by one row', None])
        assert(encoded[1] == encoded[2] == len(self.public))
        assert(le.inverse_transform(encoded) == ['setosa', None, None])
        assert('a value held by one row' not in le.categories)

    def test_unknown_value_raises_when_not_nullable(self):
        le = LabelTransformer(nullable=False, categories=self.public)
        le.fit(labels_orig)
        try:
            le.transform(['a value held by one row'])
            assert(False)
        except ValueError:
            pass

    def test_invalid_category_lists_are_rejected(self):
        for bad in ([], ['a', 'a'], ['a', None], ['a', np.nan]):
            try:
                LabelTransformer(categories=bad)
                assert(False)
            except ValueError:
                pass

    def test_default_behaviour_is_unchanged(self):
        le = LabelTransformer()
        le.fit(labels_orig)
        assert(le.categories == ['setosa', 'versicolor', 'virginica', None])
