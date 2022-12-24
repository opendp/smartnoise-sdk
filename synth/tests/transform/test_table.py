import seaborn as sns
import numpy as np
import pandas as pd
import pytest

from snsynth.transform import *
from snsynth.transform.identity import IdentityTransformer

iris = sns.load_dataset('iris')
print(iris.describe())
print(iris.head())

iris = [tuple([c for c in t[1:]]) for t in iris.itertuples()]

labels_orig = [row[4] for row in iris]
sepal_orig = [row[0] for row in iris]

sepal_large = [np.exp(v) for v in sepal_orig]

pums_csv_path = "../datasets/PUMS_null.csv"
pums = pd.read_csv(pums_csv_path, index_col=None) # in datasets/
pums.drop(['pid'], axis=1, inplace=True)
pums_categories = list(pums.columns)
pums_categories.remove('income')
pums_continuous = ['income']

class TestTableTransform:
    def test_needs_fit(self):
        tt = TableTransformer([
            MinMaxTransformer(),
            LogTransformer(),
            ChainTransformer([
                BinTransformer(),
                LabelTransformer(),
                OneHotEncoder()
            ]),
            LogTransformer(),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
        ])
        assert(not tt.fit_complete)
        tt.fit(iter(iris), epsilon=2.0)
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
        assert(tt.odometer.spent == (0.0, 0.0))
        iris_encoded = tt.transform(iris)
        iris_decoded = tt.inverse_transform(iris_encoded)
        for row_num in np.arange(5) * 2:
            row = iris[row_num]
            for idx, v in enumerate(row):
                if isinstance(v, float):
                    assert(np.isclose(v, iris_decoded[row_num][idx]))
                else:
                    assert(iris_decoded[row_num][idx] == v)
    def test_nullable_pandas_round_trip_gan(self):
        tt = TableTransformer.from_pandas(
            pums, 'gan', nullable=True,
            categorical_columns=pums_categories,
            continuous_columns=pums_continuous)
        assert(sum([1 if isinstance(x, MinMaxTransformer) else 0 for x in tt.transformers]) == 1)
        tt.fit(pums, epsilon=4.0)
        pums_encoded = tt.transform(pums)
        pums_decoded = tt.inverse_transform(pums_encoded)
        pums_decoded_iter = [tuple([c for c in t[1:]]) for t in pums_decoded.itertuples()]
        pums_iter = [tuple([c for c in t[1:]]) for t in pums.itertuples()]
        for a, b in zip(pums_iter, pums_decoded_iter):
            assert(all([x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(a, b)]))
    def test_nullable_pandas_round_trip_cube(self):
        tt = TableTransformer.from_pandas(
            pums, 'cube', nullable=True,
            categorical_columns=pums_categories,
            continuous_columns=pums_continuous)
        assert(sum([1 if isinstance(x, BinTransformer) else 0 for x in tt.transformers]) == 1)
        tt.fit(pums, epsilon=4.0)
        pums_encoded = tt.transform(pums)
        pums_decoded = tt.inverse_transform(pums_encoded)
        pums_decoded_iter = [tuple([c for c in t[1:]]) for t in pums_decoded.itertuples()]
        pums_iter = [tuple([c for c in t[1:]]) for t in pums.itertuples()]
        for a, b in zip(pums_iter, pums_decoded_iter):
            # for bins, don't check the continuous column
            a = [x if i != 4 else 1 for i, x in enumerate(a)]
            b = [x if i != 4 else 1 for i, x in enumerate(b)]
            assert(all([x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(a, b)]))
    def test_nullable_iter_round_trip_gan(self):
        tt = TableTransformer.from_pandas(
            pums, 'gan', nullable=True, 
            categorical_columns=pums_categories,
            continuous_columns=pums_continuous)
        assert(sum([1 if isinstance(x, MinMaxTransformer) else 0 for x in tt.transformers]) == 1)
        pums_iter = [tuple([c if not np.isnan(c) else None for c in t[1:]]) for t in pums.itertuples()]
        tt.fit(pums_iter, epsilon=4.0)
        pums_encoded = tt.transform(pums_iter)
        pums_decoded = tt.inverse_transform(pums_encoded)
        for a, b in zip(pums_iter, pums_decoded):
            assert(all([x == y or (x is None and y is None) for x, y in zip(a, b)]))
    def test_nullable_iter_round_trip_cube(self):
        tt = TableTransformer.from_pandas(
            pums, 'cube', nullable=True,
            categorical_columns=pums_categories,
            continuous_columns=pums_continuous)
        assert(sum([1 if isinstance(x, BinTransformer) else 0 for x in tt.transformers]) == 1)
        pums_iter = [tuple([c if not np.isnan(c) else None for c in t[1:]]) for t in pums.itertuples()]
        tt.fit(pums_iter, epsilon=4.0)
        pums_encoded = tt.transform(pums_iter)
        pums_decoded = tt.inverse_transform(pums_encoded)
        for a, b in zip(pums_iter, pums_decoded):
            # for bins, don't check the continuous column
            a = [x if i != 4 else 1 for i, x in enumerate(a)]
            b = [x if i != 4 else 1 for i, x in enumerate(b)]
            assert(all([x == y or (x is None and y is None) 
            for x, y in zip(a, b)]))
    def test_nullable_with_standard_scaler(self):
        tt = TableTransformer([
            StandardScaler(nullable=True),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
            StandardScaler(nullable=True),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
        ])
        tt.fit(pums, epsilon=4.0)
        pums_encoded = tt.transform(pums)
        pums_decoded = tt.inverse_transform(pums_encoded)
        income = pums_decoded['income'].mean()
        assert(income > 25000 and income < 45000)
    def test_empty_transformer(self):
        tt = TableTransformer([])
        tt.fit(pums, epsilon=4.0)
        pums_encoded = tt.transform(pums)
        pums_decoded = tt.inverse_transform(pums_encoded)
        assert(pums_decoded.equals(pums))
    def test_empty_transformer_numpy(self):
        tt = TableTransformer([])
        tt.fit(pums, epsilon=4.0)
        pums_np = pums.to_numpy()
        pums_encoded = tt.transform(pums_np)
        pums_decoded = tt.inverse_transform(pums_encoded)
        pairs = list(zip(pums_np.reshape(-1), pums_decoded.reshape(-1)))
        assert(all([x == y or (np.isnan(x) and np.isnan(y)) for x, y in pairs]))
    def test_empty_transformer_iter(self):
        tt = TableTransformer([])
        tt.fit(pums, epsilon=4.0)
        pums_iter = [tuple([c for c in t[1:]]) for t in pums.itertuples()]
        pums_encoded = tt.transform(pums_iter)
        pums_decoded = tt.inverse_transform(pums_encoded)
        for a, b in zip(pums_iter, pums_decoded):
            assert(all([x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(a, b)]))
    def test_anon_id(self):
        pums = pd.read_csv(pums_csv_path) # load with pid
        tt = TableTransformer([
            StandardScaler(nullable=True),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
            StandardScaler(nullable=True),
            ChainTransformer([LabelTransformer(), OneHotEncoder()]),
            AnonymizationTransformer('ssn'),
        ])
        tt.fit(pums, epsilon=4.0)
        pums_encoded = tt.transform(pums)
        pums_decoded = tt.inverse_transform(pums_encoded)
        income = pums_decoded['income'].mean()
        assert(income > 25000 and income < 45000)
    def test_anon_all_but_one(self):
        pums = pd.read_csv(pums_csv_path) # load with pid
        tt = TableTransformer([
            AnonymizationTransformer('ssn'),
            AnonymizationTransformer('date_time'),
            AnonymizationTransformer('address'),
            AnonymizationTransformer('name'),
            StandardScaler(nullable=True),
            AnonymizationTransformer('email'),
            AnonymizationTransformer('ssn')
        ])
        tt.fit(pums, epsilon=4.0)
        pums_encoded = tt.transform(pums)
        assert(len(pums_encoded[5]) == 2)
        pums_decoded = tt.inverse_transform(pums_encoded)
        income = pums_decoded['income'].mean()
        assert(income > 25000 and income < 45000)
        assert(len(pums.columns == len(pums_decoded.columns)))
    def test_anon_all(self):
        tt = TableTransformer([
            AnonymizationTransformer('ssn'),
            AnonymizationTransformer('date_time'),
            AnonymizationTransformer('address'),
            AnonymizationTransformer('name'),
            AnonymizationTransformer('ssn'),
            AnonymizationTransformer('ssn'),
        ])
        with pytest.warns(UserWarning):
            tt.fit(pums, epsilon=4.0)
        assert(tt.output_width == 0)
        pums_encoded = tt.transform(pums)
        assert(len(pums_encoded[5]) == 0)
        pums_decoded = tt.inverse_transform(pums_encoded)
        assert(len(pums_decoded.columns) == len(pums.columns))

    def test_create_tuples_with_constraints_only(self):
        tuples = [(0, 8.15, "cat_1"), (1, 3.1415, "cat_2")]
        constraints = {
            0: "ordinal",
            1: "drop",
            2: AnonymizationTransformer(lambda: "cat_3"),
        }

        tt = TableTransformer.create(tuples, constraints=constraints)
        tt.fit(tuples)
        transformed = tt.transform(tuples)
        assert len(transformed) == len(tuples)

        inversed = tt.inverse_transform(transformed)
        assert len(inversed) == len(tuples)
        assert len(inversed[0]) == 2
        assert inversed[0][0] == tuples[0][0] and inversed[0][1] == "cat_3"
        assert inversed[1][0] == tuples[1][0] and inversed[1][1] == "cat_3"

    def test_create_dataframe_with_constraints_only(self):
        columns = ["a", "b", "c"]
        df = pd.DataFrame(
            data=[(0, 8.15, "cat_1"), (1, 3.1415, "cat_2")], columns=columns
        )
        constraints = {
            "a": "ordinal",
            "b": "drop",
            "c": AnonymizationTransformer(lambda: "cat_3"),
        }

        tt = TableTransformer.create(df, constraints=constraints)
        tt.fit(df)
        transformed = tt.transform(df)
        assert len(transformed) == df.shape[0]

        inversed = tt.inverse_transform(transformed)
        assert isinstance(inversed, pd.DataFrame)
        assert inversed.shape[0] == df.shape[0]
        assert inversed.shape[1] == 2
        assert inversed.at[0, "a"] == df.at[0, "a"] and inversed.at[0, "c"] == "cat_3"
        assert inversed.at[1, "a"] == df.at[1, "a"] and inversed.at[1, "c"] == "cat_3"

    def test_create_with_constraint_and_column_list(self):
        tuples = [(0, "cat_1"), (1, "cat_2")]

        tt = TableTransformer.create(
            tuples, ordinal_columns=[0], constraints={1: "drop"}
        )
        tt.fit(tuples)
        transformed = tt.transform(tuples)
        assert len(transformed) == len(tuples)

        inversed = tt.inverse_transform(transformed)
        assert len(inversed) == len(tuples)
        assert len(inversed[0]) == 1
        assert inversed[0][0] == tuples[0][0] and inversed[1][0] == tuples[1][0]

    def test_drop_all(self):
        tuples = [(0, 8.15), (1, 3.1415)]

        tt = TableTransformer.create(tuples, constraints={0: "drop", 1: "drop"})
        assert tt.fit_complete
        assert tt.output_width == 0
        transformed = tt.transform(tuples)
        assert len(transformed) == len(tuples)
        assert all(v == () for v in transformed)

        inversed = tt.inverse_transform(transformed)
        assert len(inversed) == len(tuples)
        assert all(v == () for v in inversed)

    def test_invalid_constraint(self):
        with pytest.raises(ValueError, match="constraint.*?invalid"):
            TableTransformer.create([(0, 0)], constraints={0: None})
