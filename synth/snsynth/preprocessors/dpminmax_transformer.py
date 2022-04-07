# standard scalar transform of continuous variable

from collections import namedtuple

import numpy as np
import pandas as pd
from snsynth.preprocessors.data_transformer import BaseTransformer

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    [
        "column_name",
        "column_type",
        "transform",
        "transform_aux",
        "output_info",
        "output_dimensions",
    ],
)


# from: http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf
def quantile(vals, alpha, epsilon, lower, upper):
    k = len(vals)
    vals = [lower if v < lower else upper if v > upper else v for v in vals]
    vals = sorted(vals)
    Z = [lower] + vals + [upper]
    Z = [-lower + v for v in Z]  # shift right to be 0 bounded
    y = [
        (Z[i + 1] - Z[i]) * np.exp(-epsilon * np.abs(i - alpha * k))
        for i in range(len(Z) - 1)
    ]
    y_sum = sum(y)
    p = [v / y_sum for v in y]
    idx = np.random.choice(range(k + 1), 1, False, p)[0]
    v = np.random.uniform(Z[idx], Z[idx + 1])
    return v + lower


class MinMaxScaler(object):
    def __init__(self, epsilon, feature_range=(-1, 1), clip=False):
        self.feature_range = feature_range
        self.clip = clip
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.min_ = None
        self.epsilon = epsilon

    def fit(self, X, lower, upper):
        try:
            self.data_min_ = quantile(X, 0, self.epsilon / 2, lower, upper)
        except ValueError as e:
            print(f"error {e} occurs when calculating dp min!")
            self.data_min_ = 0
        try:
            self.data_max_ = quantile(X, 1, self.epsilon / 2, lower, upper)
        except ValueError as e:
            print(f"error {e} occurs when calcuating dp max!")
            self.data_max_ = 1000

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
            self.data_max_ - self.data_min_
        )
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_

    def transform(self, X):
        X = X * self.scale_
        X = X + self.min_
        if self.clip:
            np.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        return X

    def inverse_transform(self, X):
        X -= self.min_
        X /= self.scale_
        return X


class DPMinMaxTransformer(BaseTransformer):

    """Data Transformer.
    Based on CTGAN's transformer https://github.com/sdv-dev/CTGAN/blob/master/ctgan/data_transformer.py.
    Model continuous columns with a DPMinMax and normalized to a scalar [-1, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def _fit_continuous(self, column_name, raw_column_data, epsilon, lower, upper):
        """Fit DP Standard Scaler for continuous column."""
        scaler = MinMaxScaler(feature_range=(-1, 1), epsilon=epsilon)
        attempt = 0
        while attempt < 5:
            try:
                scaler.fit(raw_column_data, lower, upper)
                break
            except ValueError as e:

                attempt += 1
                print(f"{column_name} has Error {e}, attempts {attempt}")
        # scaler.fit(raw_column_data.reshape(-1, 1))

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=scaler,
            transform_aux=None,
            output_info=[SpanInfo(1, "tanh")],
            output_dimensions=1,
        )

    def fit(
        self, raw_data, discrete_columns=tuple(), continuous_columns_lower_upper={}
    ):
        """Fit GMM for continuous columns and One hot encoder for discrete columns.

        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        number_of_con_col = len(continuous_columns_lower_upper.keys())

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)
        else:
            self.dataframe = True

        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            raw_column_data = raw_data[column_name].values
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(column_name, raw_column_data)
            else:
                print(column_name)
                column_transform_info = self._fit_continuous(
                    column_name,
                    raw_column_data,
                    self.epsilon / number_of_con_col,
                    continuous_columns_lower_upper[column_name][0],
                    continuous_columns_lower_upper[column_name][1],
                )

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    #  print (f"column transform info list is {self._column_transform_info_list}")

    def _transform_continuous(self, column_transform_info, raw_column_data):

        scaler = column_transform_info.transform
        normalized_values = scaler.transform(raw_column_data)

        return [normalized_values]

    def _inverse_transform_continuous(
        self, column_transform_info, column_data, sigmas, st
    ):
        scaler = column_transform_info.transform
        selected_normalized_value = column_data[:, 0]

        column = scaler.inverse_transform(selected_normalized_value)

        return column
