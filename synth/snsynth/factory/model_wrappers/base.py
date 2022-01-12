"""Base Class for tabular models."""

import logging
import pickle
from warnings import warn

import numpy as np
import pandas as pd


class BaseDPTabularModel:
    """Base factory class wrapper for all smartnoise models.
    The ``BaseDPTabularModel`` class defines the common API that all the
    DPTabularModels need to implement, as well as common functionality.
    Args:
        field_names (list[str]):
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types (dict[str, dict]):
            Dictionary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        field_transformers (dict[str, str]):
            TODO
        primary_key (str):
            Name of the field which is the primary key of the table.
        constraints (list[Constraint, dict]):
            List of Constraint objects or dicts.
        table_metadata (dict or metadata.Table):
            TODO 
        rounding (int, str or None):
            Define rounding scheme for ``NumericalTransformer``. If set to an int, values
            will be rounded to that number of decimal places. If ``None``, values will not
            be rounded. If set to ``'auto'``, the transformer will round to the maximum number
            of decimal places detected in the fitted data. Defaults to ``'auto'``.
        min_value (int, str or None):
            Specify the minimum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be greater than or equal to it. If the string ``'auto'``
            is given, the minimum will be the minimum value seen in the fitted data. If ``None``
            is given, there won't be a minimum. Defaults to ``'auto'``.
        max_value (int, str or None):
            Specify the maximum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be less than or equal to it. If the string ``'auto'``
            is given, the maximum will be the maximum value seen in the fitted data. If ``None``
            is given, there won't be a maximum. Defaults to ``'auto'``.
    """

    _DTYPE_TRANSFORMERS = None

    _metadata = None

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None, 
                 rounding=None,max_value=None,min_value=None):
        self.field_names=field_names
        self.primary_key=primary_key
        self.field_types=field_types
        self.field_transformers=field_transformers
        self.anonymize_fields=anonymize_fields
        self.constraints=constraints
        self.table_metadata=table_metadata
        self.rounding=rounding,
        self.max_value=max_value,
        self.min_value=min_value

    def fit(self, data):
        """Fit this model to the data.
        If the table metadata has not been given, learn it from the data.
        Args:
            data (pandas.DataFrame or str):
                Data to fit the model to. It can be passed as a
                ``pandas.DataFrame`` or as an ``str``.
                If an ``str`` is passed, it is assumed to be
                the path to a CSV file which can be loaded using
                ``pandas.read_csv``.
        """
        if isinstance(data, pd.DataFrame):
            data = data.reset_index(drop=True)

        if not self._metadata_fitted:
            self._metadata.fit(data)

        self._num_rows = len(data)

        self._fit(data) 

        # TODO: Replace?
        # transformed = self._metadata.transform(data)

    def sample(self, num_rows=None):
        """Sample rows from this table.
        Args:
            num_rows (int):
                Number of rows to sample. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if num_rows is None:
            num_rows = self._num_rows
        return self._sample(num_rows)

    def save(self, path):
        """Save this model instance to the given path using pickle.
        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a TabularModel instance from a given path.
        Args:
            path (str):
                Path from which to load the instance.
        Returns:
            TabularModel:
                The loaded tabular model.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)