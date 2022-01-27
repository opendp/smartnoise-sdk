"""Transformer for data that contains Null values."""

import warnings

import numpy as np
import pandas as pd

IRREVERSIBLE_WARNING = (
    'Replacing nulls with existing value without `null_column`, which is not reversible. '
    'Use `null_column=True` to ensure that the transformation is reversible.'
)


class NullTransformer():
    """Transformer for data that contains Null values.

    Args:
        fill_value (object or None):
            Value to replace nulls, or strategy to compute the value, which can
            be ``mean`` or ``mode``. If ``None`` is given, the ``mean`` or ``mode``
            strategy will be applied depending on whether the input data is numerical
            or not. Defaults to `None`.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        copy (bool):
            Whether to create a copy of the input data or modify it destructively.
    """

    nulls = None
    _null_column = None
    _fill_value = None

    def __init__(self, fill_value=None, null_column=None, copy=False):
        self.fill_value = fill_value
        self.null_column = null_column
        self.copy = copy

    def creates_null_column(self):
        """Indicate whether this transformer creates a null column on transform.

        Returns:
            bool:
                Whether a null column is created on transform.
        """
        return bool(self._null_column)

    def _get_fill_value(self, data, null_values):
        """Get the fill value to use for the given data.

        Args:
            data (pd.Series):
                The data that is being transformed.
            null_values (np.array):
                Array of boolean values that indicate which values in the
                input data are nulls.

        Return:
            object:
                The fill value that needs to be used.
        """
        fill_value = self.fill_value

        if fill_value in (None, 'mean', 'mode') and null_values.all():
            return 0

        if fill_value is None:
            if pd.api.types.is_numeric_dtype(data):
                fill_value = 'mean'
            else:
                fill_value = 'mode'

        if fill_value == 'mean':
            return data.mean()

        if fill_value == 'mode':
            return data.mode(dropna=True)[0]

        return fill_value

    def fit(self, data):
        """Fit the transformer to the data.

        Evaluate if the transformer has to create the null column or not.

        Args:
            data (pandas.Series):
                Data to transform.
        """
        null_values = data.isna().to_numpy()
        self.nulls = null_values.any()

        self._fill_value = self._get_fill_value(data, null_values)

        if self.null_column is None:
            self._null_column = self.nulls
        else:
            self._null_column = self.null_column

    def transform(self, data):
        """Replace null values with the indicated fill_value.

        If required, create the null indicator column.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        isna = data.isna()
        if isna.any():
            if not self._null_column and self._fill_value in data.to_numpy():
                warnings.warn(IRREVERSIBLE_WARNING)

            if not self.copy:
                data[isna] = self._fill_value
            else:
                data = data.fillna(self._fill_value)

        if self._null_column:
            return pd.concat([data, isna.astype(np.float64)], axis=1).to_numpy()

        return data.to_numpy()

    def reverse_transform(self, data):
        """Restore null values to the data.

        If a null indicator column was created during fit, use it as a reference.
        Otherwise, replace all instances of ``fill_value`` that can be found in
        data.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if self._null_column:
            if self.nulls:
                isna = data[:, 1] > 0.5

            data = data[:, 0]
            if self.copy:
                data = data.copy()

        elif self.nulls:
            isna = self._fill_value == data

        data = pd.Series(data)

        if self.nulls and isna.any():
            data.loc[isna] = np.nan

        return data