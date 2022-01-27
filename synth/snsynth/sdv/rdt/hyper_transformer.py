"""Hyper transformer module."""

import warnings
from collections import defaultdict
from copy import deepcopy

import yaml

from snsynth.sdv.rdt.transformers.categorical import LabelEncodingTransformer, OneHotEncodingTransformer
from snsynth.sdv.rdt.transformers import get_transformer_instance

from snsynth.sdv.rdt.errors import NotFittedError

class HyperTransformer:
    """HyperTransformer class.
    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.
    Args:
        field_transformers (dict or None):
            Dict used to overwrite the transformer used for a field. If no transformer is
            specified for a field, a default transformer is selected. The keys are fields
            which can be defined as a string of the column name or a tuple of multiple column
            names. Keys can also specify transformers for fields derived by other transformers.
            This can be done by concatenating the name of the original field to the output name
            using ``.`` as a separator (eg. {field_name}.{transformer_output_name}).
        field_data_types (dict or None):
            Dict mapping field names to their data types. If not provided, the data type is
            inferred using the column's Pandas ``dtype``.
        default_data_type_transformers (dict or None):
            Dict used to overwrite the default transformer for a data type. The keys are
            data types and the values are Transformers or Transformer instances.
        copy (bool):
            Whether to make a copy of the input data or not. Defaults to ``True``.
        transform_output_types (list or None):
            List of acceptable data types for the output of the ``transform`` method.
            If ``None``, only ``numerical`` types will be considered acceptable.
    Example:
        Create a simple ``HyperTransformer`` instance that will decide which transformers
        to use based on the fit data ``dtypes``.
        >>> ht = HyperTransformer()
        Create a ``HyperTransformer`` passing a dict mapping fields to data types.
        >>> field_data_types = {
        ...     'a': 'categorical',
        ...     'b': 'numerical
        ... }
        >>> ht = HyperTransformer(field_data_types=field_data_types)
        Create a ``HyperTransformer`` passing a ``field_transformers`` dict.
        (Note: The transformers used in this example may not exist and are just used
        to illustrate the different way that a transformer can be defined for a field).
        >>> field_transformers = {
        ...     'email': EmailTransformer(),
        ...     'email.domain': EmailDomainTransformer(),
        ...     ('year', 'month', 'day'): DateTimeTransformer()
        ... }
        >>> ht = HyperTransformer(field_transformers=field_transformers)
        Create a ``HyperTransformer`` passing a dict mapping data types to transformers.
        >>> default_data_type_transformers = {
        ...     'categorical': LabelEncodingTransformer(),
        ...     'numerical': NumericalTransformer()
        ... }
        >>> ht = HyperTransformer(default_data_type_transformers=default_data_type_transformers)
    """

    # pylint: disable=too-many-instance-attributes

    _DTYPES_TO_DATA_TYPES = {
        'i': 'integer',
        'f': 'float',
        'O': 'categorical',
    }
    _DEFAULT_OUTPUT_TYPES = [
        'numerical',
        'float',
        'integer'
    ]

    @staticmethod
    def _add_field_to_set(field, field_set):
        if isinstance(field, tuple):
            field_set.update(field)
        else:
            field_set.add(field)  # noqa -> set can't use opreator

    @staticmethod
    def _field_in_set(field, field_set):
        if isinstance(field, tuple):
            return all(column in field_set for column in field)

        return field in field_set

    @staticmethod
    def _subset(input_list, other_list, not_in=False):
        return [
            element
            for element in input_list
            if (element in other_list) ^ not_in
        ]

    def _create_multi_column_fields(self):
        multi_column_fields = {}
        for field in list(self.field_data_types) + list(self.field_transformers):
            if isinstance(field, tuple):
                for column in field:
                    multi_column_fields[column] = field
        return multi_column_fields

    def _validate_field_transformers(self):
        for field in self.field_transformers:
            if self._field_in_set(field, self._specified_fields):
                raise ValueError(f'Multiple transformers specified for the field {field}. '
                                 'Each field can have at most one transformer defined in '
                                 'field_transformers.')

            self._add_field_to_set(field, self._specified_fields)

    def __init__(self, copy=True, field_data_types=None, default_data_type_transformers=None,
                 field_transformers=None, transform_output_types=None):
        self.copy = copy
        self.field_data_types = field_data_types or {}
        self.default_data_type_transformers = default_data_type_transformers or {}
        self.field_transformers = field_transformers or {}
        self._specified_fields = set()
        self._validate_field_transformers()
        self.transform_output_types = transform_output_types or self._DEFAULT_OUTPUT_TYPES
        self._multi_column_fields = self._create_multi_column_fields()
        self._transformers_sequence = []
        self._output_columns = []
        self._input_columns = []
        self._fitted_fields = set()
        self._fitted = False
        self._transformers_tree = defaultdict(dict)

    @staticmethod
    def _field_in_data(field, data):
        all_columns_in_data = isinstance(field, tuple) and all(col in data for col in field)
        return field in data or all_columns_in_data

    def _populate_field_data_types(self, data):
        # get set of provided fields including multi-column fields
        provided_fields = set()
        for field in self.field_data_types.keys():
            self._add_field_to_set(field, provided_fields)

        for field in data:
            if field not in provided_fields:
                clean_data = data[field].dropna()
                kind = clean_data.infer_objects().dtype.kind
                self.field_data_types[field] = self._DTYPES_TO_DATA_TYPES[kind]

    def _unfit(self):
        self._transformers_sequence = []
        self._fitted = False

    def get_field_data_types(self):
        """Get the ``field_data_types`` dict.
        Returns:
            dict:
                Mapping of fields to their data types. Fields can be defined as a string
                representing a column name or a tuple of multiple column names.
        """
        return self.field_data_types

    def update_field_data_types(self, field_data_types):
        """Update the ``field_data_types`` dict.
        Args:
            field_data_types (dict):
                Mapping of fields to their data types. Fields can be defined as a string
                representing a column name or a tuple of multiple column names. It will
                update the existing ``field_data_types`` values. Calling this method will
                require ``fit`` to be run again.
        """
        self.field_data_types.update(field_data_types)
        self._unfit()

    def get_default_data_type_transformers(self):
        """Get the ``default_data_type_transformer`` dict.
        Returns:
            dict:
                The ``default_data_type_transformers`` dictionary. The keys are
                data types and the values are Transformers or Transformer instances.
        """
        return self.default_data_type_transformers

    def update_default_data_type_transformers(self, new_data_type_transformers):
        """Update the ``default_data_type_transformer`` dict.
        Args:
            new_data_type_transformers (dict):
                Dict mapping data types to the default transformer class or instance to use for
                them. This dict does not need to contain an entry for every data type. It will be
                used to overwrite the existing defaults. Calling this method will require ``fit``
                to be run again.
        """
        self.default_data_type_transformers.update(new_data_type_transformers)
        self._unfit()

    def set_first_transformers_for_fields(self, field_transformers):
        """Set the first transformer to use for certain fields.
        Args:
            field_transformers (dict):
                Dict mapping fields to a transformer class name or instance. This transformer will
                be the first used on that field when the ``HyperTransformer`` calls ``transform``.
                The fields or keys can be defined as strings representing a single column name, or
                tuples of strings representing multiple column names. Calling this method will
                require ``fit`` to be run again.
        """
        self.field_transformers.update(field_transformers)
        self._unfit()

    def get_transformer(self, field):
        """Get the transformer instance used for a field.
        Args:
            field (str or tuple):
                String representing a column name or a tuple of multiple column names.
        Returns:
            Transformer:
                Transformer instance used on the specified field during ``transform``.
        """
        if not self._fitted:
            raise NotFittedError

        return self._transformers_tree[field].get('transformer', None)

    def get_output_transformers(self, field):
        """Return dict mapping output columns of field to transformers used on them.
        Args:
            field (str or tuple):
                String representing a column name or a tuple of multiple column names.
        Returns:
            dict:
                Dictionary mapping the output names of the columns created after transforming the
                specified field, to the transformer instances used on them.
        """
        if not self._fitted:
            raise NotFittedError

        next_transformers = {}
        for output in self._transformers_tree[field].get('outputs', []):
            next_transformers[output] = self._transformers_tree[output].get('transformer', None)

        return next_transformers

    def get_final_output_columns(self, field):
        """Return list of all final output columns related to a field.
        The ``HyperTransformer`` will figure out which transformers to use on a field during
        ``transform``. If the outputs are not of an acceptable data type, they will also go
        through transformations. This method finds all the output columns that are of an
        acceptable final data type that originated from the specified field.
        Args:
            field (str or tuple):
                String representing a column name or a tuple of multiple column names.
        Returns:
            list:
                List of output column names that were created as a by-product of the specified
                field.
        """
        if not self._fitted:
            raise NotFittedError

        final_outputs = []
        outputs = self._transformers_tree[field].get('outputs', []).copy()
        while len(outputs) > 0:
            output = outputs.pop()
            if output in self._transformers_tree:
                outputs.extend(self._transformers_tree[output].get('outputs', []))
            else:
                final_outputs.append(output)

        return final_outputs

    def get_transformer_tree_yaml(self):
        """Return yaml representation of transformers tree.
        After running ``fit``, a sequence of transformers is created to run each original column
        through. The sequence can be thought of as a tree, where each node is a field and the
        transformer used on it, and each neighbor is an output from that transformer. This method
        returns a YAML representation of this tree.
        Returns:
            string:
                YAML object representing the tree of transformers created during ``fit``. It has
                the following form:
                field1:
                    transformer: ExampleTransformer instance
                    outputs: [field1.out1, field1.out2]
                field1.out1:
                    transformer: CategoricalTransformer instance
                    outputs: [field1.out1.value]
                field1.out2:
                    transformer: CategoricalTransformer instance
                    outputs: [field1.out2.value]
        """
        modified_tree = deepcopy(self._transformers_tree)
        for field in modified_tree:
            class_name = modified_tree[field]['transformer'].__class__.__name__
            modified_tree[field]['transformer'] = class_name

        return yaml.safe_dump(modified_tree)

    def _fit_field_transformer(self, data, field, transformer):
        """Fit a transformer to its corresponding field.
        This method fits a transformer to the specified field which can be a column
        name or tuple of column names. If the transformer outputs fields that aren't
        ML ready, then this method recursively fits transformers to their outputs until
        they are. This method keeps track of which fields are temporarily created by
        transformers as well as which fields will be part of the final output from ``transform``.
        Args:
            data (pandas.DataFrame):
                Data to fit the transformer to.
            field (str or tuple):
                Name of column or tuple of columns in data that will be transformed
                by the transformer.
            transformer (Transformer):
                Instance of transformer class that will fit the data.
        """
        transformer = get_transformer_instance(transformer)
        transformer.fit(data, field)
        self._add_field_to_set(field, self._fitted_fields)
        self._transformers_sequence.append(transformer)
        data = transformer.transform(data)

        output_types = transformer.get_output_types()
        next_transformers = transformer.get_next_transformers()
        self._transformers_tree[field]['transformer'] = transformer
        self._transformers_tree[field]['outputs'] = list(output_types)
        for (output_name, output_type) in output_types.items():
            output_field = self._multi_column_fields.get(output_name, output_name)
            
            if output_name not in self._output_columns:
                self._output_columns.append(output_name)

        return data

    def _validate_all_fields_fitted(self):
        non_fitted_fields = self._specified_fields.difference(self._fitted_fields)
        if non_fitted_fields:
            warnings.warn('The following fields were specified in the input arguments but not'
                          + f'found in the data: {non_fitted_fields}')

    def fit(self, data):
        """Fit the transformers to the data.
        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        self._input_columns = list(data.columns)
        self._populate_field_data_types(data)

        # Loop through field_transformers that are first level
        for field in self.field_transformers:
            if self._field_in_data(field, data):
                data = self._fit_field_transformer(data, field, self.field_transformers[field])

        for (field, data_type) in self.field_data_types.items():
            if not self._field_in_set(field, self._fitted_fields):
                transformer = self.default_data_type_transformers[data_type]

                data = self._fit_field_transformer(data, field, transformer)

        self._validate_all_fields_fitted()
        self._fitted = True

    def transform(self, data):
        """Transform the data.
        If ``self.copy`` is ``True`` make a copy of the input data to avoid modifying it.
        Args:
            data (pandas.DataFrame):
                Data to transform.
        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        if not self._fitted:
            raise NotFittedError

        unknown_columns = self._subset(data.columns, self._input_columns, not_in=True)
        if self.copy:
            data = data.copy()

        for transformer in self._transformers_sequence:
            data = transformer.transform(data, drop=False)

        transformed_columns = self._subset(self._output_columns, data.columns)
        return data.reindex(columns=unknown_columns + transformed_columns)

    def fit_transform(self, data):
        """Fit the transformers to the data and then transform it.
        Args:
            data (pandas.DataFrame):
                Data to transform.
        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        """Revert the transformations back to the original values.
        Args:
            data (pandas.DataFrame):
                Data to revert.
        Returns:
            pandas.DataFrame:
                reversed data.
        """
        if not self._fitted:
            raise NotFittedError

        unknown_columns = self._subset(data.columns, self._output_columns, not_in=True)
        for transformer in reversed(self._transformers_sequence):
            data = transformer.reverse_transform(data, drop=False)

        reversed_columns = self._subset(self._input_columns, data.columns)

        return data.reindex(columns=unknown_columns + reversed_columns)