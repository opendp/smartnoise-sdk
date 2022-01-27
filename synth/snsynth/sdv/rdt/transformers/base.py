"""BaseTransformer module.
Based heavily on SDV's base transformer module: https://github.com/sdv-dev/RDT/blob/master/rdt/transformers/base.py
"""
import abc

import pandas as pd


class BaseTransformer:
    """Base class for all transformers.
    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``_fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """

    INPUT_TYPE = None
    OUTPUT_TYPES = None
    DETERMINISTIC_TRANSFORM = None
    DETERMINISTIC_REVERSE = None
    COMPOSITION_IS_IDENTITY = None
    NEXT_TRANSFORMERS = None

    columns = None
    column_prefix = None
    output_columns = None

    @classmethod
    def get_subclasses(cls):
        """Recursively find subclasses of this Baseline.
        Returns:
            list:
                List of all subclasses of this class.
        """
        subclasses = []
        for subclass in cls.__subclasses__():
            if abc.ABC not in subclass.__bases__:
                subclasses.append(subclass)

            subclasses += subclass.get_subclasses()

        return subclasses

    @classmethod
    def get_input_type(cls):
        """Return the input type supported by the transformer.
        Returns:
            string:
                Accepted input type of the transformer.
        """
        return cls.INPUT_TYPE

    def _add_prefix(self, dictionary):
        if not dictionary:
            return None

        output = {}
        for output_columns, output_type in dictionary.items():
            output[f'{self.column_prefix}.{output_columns}'] = output_type

        return output

    def get_output_types(self):
        """Return the output types produced by this transformer.
        Returns:
            dict:
                Mapping from the transformed column names to the produced data types.
        """
        return self._add_prefix(self.OUTPUT_TYPES)

    def is_transform_deterministic(self):
        """Return whether the transform is deterministic.
        Returns:
            bool:
                Whether or not the transform is deterministic.
        """
        return self.DETERMINISTIC_TRANSFORM

    def is_reverse_deterministic(self):
        """Return whether the reverse transform is deterministic.
        Returns:
            bool:
                Whether or not the reverse transform is deterministic.
        """
        return self.DETERMINISTIC_REVERSE

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.
        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        return self.COMPOSITION_IS_IDENTITY

    def get_next_transformers(self):
        """Return the suggested next transformer to be used for each column.
        Returns:
            dict:
                Mapping from transformed column names to the transformers to apply to each column.
        """
        return self._add_prefix(self.NEXT_TRANSFORMERS)

    def get_input_columns(self):
        """Return list of input column names for transformer.
        Returns:
            list:
                Input column names.
        """
        return self.columns

    def get_output_columns(self):
        """Return list of column names created in ``transform``.
        Returns:
            list:
                Names of columns created during ``transform``.
        """
        return [f'{self.column_prefix}.{output}' for output in self.OUTPUT_TYPES]

    def _store_columns(self, columns, data):
        if isinstance(columns, tuple) and columns not in data:
            columns = list(columns)
        elif not isinstance(columns, list):
            columns = [columns]

        missing = set(columns) - set(data.columns)
        if missing:
            raise KeyError(f'Columns {missing} were not present in the data.')

        self.columns = columns

    @staticmethod
    def _get_columns_data(data, columns):
        if len(columns) == 1:
            columns = columns[0]

        return data[columns]

    @staticmethod
    def _set_columns_data(data, columns_data, columns):
        if isinstance(columns_data, (pd.DataFrame, pd.Series)):
            columns_data.index = data.index

        if len(columns_data.shape) == 1:
            data[columns[0]] = columns_data
        else:
            data[columns] = columns_data

    def _build_output_columns(self, data):
        self.column_prefix = '#'.join(self.columns)
        self.output_columns = list(self.get_output_types().keys())

        # make sure none of the generated `output_columns` exists in the data
        data_columns = set(data.columns)
        while data_columns & set(self.output_columns):
            self.column_prefix += '#'
            self.output_columns = list(self.get_output_types().keys())

    def _fit(self, columns_data):
        """Fit the transformer to the data.
        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.
        """
        raise NotImplementedError()

    def fit(self, data, columns):
        """Fit the transformer to the `columns` of the `data`.
        Args:
            data (pandas.DataFrame):
                The entire table.
            columns (list):
                Column names. Must be present in the data.
        """
        self._store_columns(columns, data)

        columns_data = self._get_columns_data(data, self.columns)
        self._fit(columns_data)

        self._build_output_columns(data)

    def _transform(self, columns_data):
        """Transform the data.
        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.
        Returns:
            pandas.DataFrame or pandas.Series:
                Transformed data.
        """
        raise NotImplementedError()

    def transform(self, data, drop=True):
        """Transform the `self.columns` of the `data`.
        Args:
            data (pandas.DataFrame):
                The entire table.
            drop (bool):
                Whether or not to drop original columns.
        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        # if `data` doesn't have the columns that were fitted on, don't transform
        if any(column not in data.columns for column in self.columns):
            return data

        data = data.copy()

        columns_data = self._get_columns_data(data, self.columns)
        transformed_data = self._transform(columns_data)

        self._set_columns_data(data, transformed_data, self.output_columns)
        if drop:
            data = data.drop(self.columns, axis=1)

        return data

    def fit_transform(self, data, columns):
        """Fit the transformer to the `columns` of the `data` and then transform them.
        Args:
            data (pandas.DataFrame):
                The entire table.
            columns (list or tuple or str):
                List or tuple of column names from the data to transform.
                If only one column is provided, it can be passed as a string instead.
                If none are passed, fits on the entire dataset.
        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        self.fit(data, columns)
        return self.transform(data)

    def _reverse_transform(self, columns_data):
        """Revert the transformations to the original values.
        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to revert.
        Returns:
            pandas.DataFrame or pandas.Series:
                Reverted data.
        """
        raise NotImplementedError()

    def reverse_transform(self, data, drop=True):
        """Revert the transformations to the original values.
        Args:
            data (pandas.DataFrame):
                The entire table.
            drop (bool):
                Whether or not to drop derived columns.
        Returns:
            pandas.DataFrame:
                The entire table, containing the reverted data.
        """
        # if `data` doesn't have the columns that were transformed, don't reverse_transform
        if any(column not in data.columns for column in self.output_columns):
            return data

        data = data.copy()

        columns_data = self._get_columns_data(data, self.output_columns)
        reversed_data = self._reverse_transform(columns_data)

        self._set_columns_data(data, reversed_data, self.columns)
        if drop:
            data = data.drop(self.output_columns, axis=1)

        return data