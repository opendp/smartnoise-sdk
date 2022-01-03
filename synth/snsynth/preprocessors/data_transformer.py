# standard scalar transform of continuous variable

from collections import namedtuple

import numpy as np
import pandas as pd
from rdt.transformers import OneHotEncodingTransformer
from sklearn.mixture import BayesianGaussianMixture
from diffprivlib.models import StandardScaler

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo", ["column_name", "column_type",
                            "transform", 
                            "output_info", "output_dimensions"])

class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a DPStandardScaler and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, epsilon):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            epsilon (float):
                epsilon for DP StandardScaler preprocessor
        """
        

        self.epsilon = epsilon

    def _fit_continuous(self, column_name, raw_column_data):
        """Fit DP Standard Scaler for continuous column."""
     
        
        scaler = StandardScaler(self.epsilon)
        scaler.fit(raw_column_data.reshape(-1,1))

        

        return ColumnTransformInfo(
            column_name=column_name, column_type="continuous", transform=scaler,
            output_info=[SpanInfo(1, 'tanh')],
            output_dimensions=1)

    def _fit_discrete(self, column_name, raw_column_data):
        """Fit one hot encoder for discrete column."""
        ohe = OneHotEncodingTransformer()
        ohe.fit(raw_column_data)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type="discrete", transform=ohe,
            transform_aux=None,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data, discrete_columns=tuple()):
        """Fit DP StandardScaler for continuous columns and One hot encoder for discrete columns.

        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0

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
                column_transform_info = self._fit_discrete(
                    column_name, raw_column_data)
            else:
                column_transform_info = self._fit_continuous(
                    column_name, raw_column_data)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)
     

    def _transform_continuous(self, column_transform_info, raw_column_data):
        
        scaler = column_transform_info.transform
        normalized_values = scaler.transform(raw_column_data)
        normalized_values = normalized_values/4
        


        
        
        return [normalized_values]

    def _transform_discrete(self, column_transform_info, raw_column_data):
        ohe = column_transform_info.transform
        return [ohe.transform(raw_column_data)]

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = raw_data[[column_transform_info.column_name]].values
            if column_transform_info.column_type == "continuous":
                column_data_list += self._transform_continuous(
                    column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == "discrete"
                column_data_list += self._transform_discrete(
                    column_transform_info, column_data)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        scaler = column_transform_info.transform
        selected_normalized_value = column_data[:, 0]

        stds = np.sqrt(scaler.var_)
        print (f"stds is {stds}")
        means = scaler.mean_
        column = selected_normalized_value * 4 * stds + means




        return column

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        return ohe.reverse_transform(column_data)

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]

            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
                
            else:
                assert column_transform_info.column_type == 'discrete'
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.values

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        one_hot = column_transform_info.transform.transform(np.array([value]))[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot)
        }
