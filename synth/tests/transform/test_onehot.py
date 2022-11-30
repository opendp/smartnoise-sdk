import pytest
import seaborn as sns
from snsynth.transform.label import LabelTransformer
from snsynth.transform.onehot import OneHotEncoder

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t]) for t in iris.itertuples()]

labels_orig = [row[5] for row in iris]
sepal_orig = [row[1] for row in iris]

le = LabelTransformer()
le.fit(iris, 5)
categories = le.transform(labels_orig)

class TestOneHot:
    def test_onehot_by_val(self):
        ohe = OneHotEncoder()
        ohe.fit(categories)
        assert(ohe.output_width == 3)
        cat_encoded = ohe.transform(categories)
        cat_decoded = ohe.inverse_transform(cat_encoded)
        assert(all([a == b for a, b in zip(categories, cat_decoded)]))
    def test_onehot_by_val_fittransform(self):
        ohe = OneHotEncoder()
        cat_encoded = ohe.fit_transform(categories)
        assert(ohe.output_width == 3)
        cat_decoded = ohe.inverse_transform(cat_encoded)
        assert(all([a == b for a, b in zip(categories, cat_decoded)]))
    def test_onehot_single_val(self):
        ohe = OneHotEncoder()
        data = [0]
        ohe.fit(data)
        assert(ohe.output_width == 1)
        data_encoded = ohe.transform(data)
        assert data_encoded[0] == 1
        data_decoded = ohe.inverse_transform(data_encoded)
        assert data_decoded == data

    def test_invalid_val(self):
        ohe = OneHotEncoder()
        ohe.fit([0])
        for data in [[-1], [-0.1], [0.1], [1]]:
            with pytest.raises(ValueError, match="label.*?invalid"):
                ohe.transform(data)

    def test_not_fit(self):
        ohe = OneHotEncoder()
        with pytest.raises(ValueError, match="not.*?fit"):
            ohe.transform([0])
