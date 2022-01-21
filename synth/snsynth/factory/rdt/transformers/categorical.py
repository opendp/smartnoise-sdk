import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer

class OneHotEncodingTransformer(BaseTransformer):
    """OneHotEncoding for categorical data.
    This transformer replaces a single vector with N unique categories in it
    with N vectors which have 1s on the rows where the corresponding category
    is found and 0s on the rest.
    Null values are considered just another category.
    Args:
        error_on_unknown (bool):
            If a value that was not seen during the fit stage is passed to
            transform, then an error will be raised if this is True.
            Defaults to ``True``.
    """

    INPUT_TYPE = 'categorical'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True

    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None

    def __init__(self, error_on_unknown=True):
        self.error_on_unknown = error_on_unknown

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.
        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.
        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to prepare.
        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')

            data = data[:, 0]

        return data

    def get_output_types(self):
        """Return the output types produced by this transformer.
        Returns:
            dict:
                Mapping from the transformed column names to the produced data types.
        """
        output_types = {f'value{i}': 'float' for i in range(len(self.dummies))}

        return self._add_prefix(output_types)

    def _fit(self, data):
        """Fit the transformer to the data.
        Get the pandas `dummies` which will be used later on for OneHotEncoding.
        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to fit the transformer to.
        """
        data = self._prepare_data(data)

        null = pd.isna(data)
        self._uniques = list(pd.unique(data[~null]))
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()

        if not np.issubdtype(data.dtype, np.number):
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

    def _transform_helper(self, data):
        if self._dummy_encoded:
            coder = self._indexer
            codes = pd.Categorical(data, categories=self._uniques).codes
        else:
            coder = self._uniques
            codes = data

        rows = len(data)
        dummies = np.broadcast_to(coder, (rows, self._num_dummies))
        coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
        array = (coded == dummies).astype(int)

        if self._dummy_na:
            null = np.zeros((rows, 1), dtype=int)
            null[pd.isna(data)] = 1
            array = np.append(array, null, axis=1)

        return array

    def _transform(self, data):
        """Replace each category with the OneHot vectors.
        Args:
            data (pandas.Series, list or list of lists):
                Data to transform.
        Returns:
            numpy.ndarray:
        """
        data = self._prepare_data(data)
        array = self._transform_helper(data)

        if self.error_on_unknown:
            unknown = array.sum(axis=1) == 0
            if unknown.any():
                raise ValueError(f'Attempted to transform {list(data[unknown])} ',
                                 'that were not seen during fit stage.')

        return array

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.
        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.
        Returns:
            pandas.Series
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)

        return pd.Series(indices).map(self.dummies.__getitem__)


class LabelEncodingTransformer(BaseTransformer):
    """LabelEncoding for categorical data.
    This transformer generates a unique integer representation for each category
    and simply replaces each category with its integer value.
    Null values are considered just another category.
    Attributes:
        values_to_categories (dict):
            Dictionary that maps each integer value for its category.
        categories_to_values (dict):
            Dictionary that maps each category with the corresponding
            integer value.
    """

    INPUT_TYPE = 'categorical'
    OUTPUT_TYPES = {'value': 'integer'}
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    values_to_categories = None
    categories_to_values = None

    def _fit(self, data):
        """Fit the transformer to the data.
        Generate a unique integer representation for each category and
        store them in the `categories_to_values` dict and its reverse
        `values_to_categories`.
        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.values_to_categories = dict(enumerate(pd.unique(data)))
        self.categories_to_values = {
            category: value
            for value, category in self.values_to_categories.items()
        }

    def _transform(self, data):
        """Replace each category with its corresponding integer value.
        Args:
            data (pandas.Series):
                Data to transform.
        Returns:
            numpy.ndarray:
        """
        return pd.Series(data).map(self.categories_to_values)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.
        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.
        Returns:
            pandas.Series
        """
        data = data.clip(min(self.values_to_categories), max(self.values_to_categories))
        return data.round().map(self.values_to_categories)