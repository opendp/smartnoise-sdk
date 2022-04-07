# standard scalar transform of continuous variable

from collections import namedtuple

import numpy as np
import pandas as pd
from diffprivlib.models import StandardScaler
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


class DPSSTransformer(BaseTransformer):

    """Data Transformer.
    Based on CTGAN's transformer https://github.com/sdv-dev/CTGAN/blob/master/ctgan/data_transformer.py.
    Model continuous columns with a DPStandardScaler and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def _fit_continuous(self, column_name, raw_column_data, epsilon):
        """Fit DP Standard Scaler for continuous column."""
        scaler = StandardScaler(epsilon=epsilon)
        scaler.fit(raw_column_data.reshape(-1, 1))

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

        if continuous_columns_lower_upper and len(continuous_columns_lower_upper) > 0:
            number_of_con_col = len(continuous_columns_lower_upper.keys())
        else:
            number_of_con_col = raw_data.shape[1] - len(discrete_columns)

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
                column_transform_info = self._fit_continuous(
                    column_name, raw_column_data, self.epsilon / number_of_con_col
                )

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, raw_column_data):

        scaler = column_transform_info.transform
        normalized_values = scaler.transform(raw_column_data)
        normalized_values = normalized_values / 4

        return [normalized_values]

    def _inverse_transform_continuous(
        self, column_transform_info, column_data, sigmas, st
    ):
        scaler = column_transform_info.transform
        selected_normalized_value = column_data[:, 0]

        stds = np.sqrt(scaler.var_)
        # print(f"stds is {stds}")
        means = scaler.mean_
        column = selected_normalized_value * 4 * stds + means

        return column
