import pandas as pd
import numpy as np
import warnings

from snsql.sql.odometer import OdometerHeterogeneous
from snsql.sql.privacy import Privacy

from .anonymization import AnonymizationTransformer
from .drop import DropTransformer
from snsynth.transform.type_map import TypeMap

class TableTransformer:
    """Transforms a table of data.

    :param transformers: a list of transformers, one per column
    :param odometer: an optional odometer to use to track privacy spent when fitting the data
    """
    def __init__(self, transformers=[], *ignore, odometer=None):
        # one transformer per input column
        self.transformers = transformers
        if self.fit_complete:
            self.output_width = sum([t.output_width for t in self.transformers])
        else:
            self.output_width = 0
        if odometer is None:
            self.odometer = OdometerHeterogeneous(privacy=Privacy())
        else:
            self.odometer = odometer

        self._columns = None # will be automatically set if pandas
        self._dtype = None # will be automatically set if numpy
        self._dropped_column_indices = set()  # used if a pd.DataFrame and at least one DropTransformer are provided

    @property
    def fit_complete(self):
        """Returns True if the transformer has been fit."""
        return all([t.fit_complete for t in self.transformers])
    @property
    def needs_epsilon(self):
        """Returns True if the transformer needs to spend privacy budget when fitting."""
        return any([t.needs_epsilon for t in self.transformers])
    @property
    def cardinality(self):
        """Returns the cardinality of each output column.  Returns None for continuous columns."""
        cards = []
        for t in self.transformers:
            if isinstance(t, DropTransformer) or isinstance(t, AnonymizationTransformer):
                continue # don't include cardinality for the unbounded transformers
            for c in t.cardinality:
                cards.append(c)
        return cards
    def allocate_privacy_budget(self, epsilon, odometer):
        n_with_epsilon = sum([1 for t in self.transformers if t.needs_epsilon])
        if n_with_epsilon == 0:
            return
        else:
            for transformer in self.transformers:
                if transformer.needs_epsilon:
                    transformer.allocate_privacy_budget(epsilon / n_with_epsilon, odometer)
    def fit(self, data, *ignore, epsilon=None):
        """Fits the transformer to the data.

        :param data: a table represented as a list of tuples, a numpy.ndarray, or a pandas DataFrame
        :param epsilon: the privacy budget to spend fitting the data
        """
        if self.transformers == []:
            self._fit_finish()
        if epsilon is not None and epsilon > 0.0:
            self.allocate_privacy_budget(epsilon, self.odometer)
        if isinstance(data, pd.DataFrame):
            self._columns = list(data.columns)
            data = [tuple([c for c in t[1:]]) for t in data.itertuples()]
        self._dropped_column_indices = set()
        for t in self.transformers:
            t._clear_fit()
        for row in data:
            self._fit(row)
        for t in self.transformers:
            t._fit_finish()
        self._fit_finish()
        if self.output_width == 0:
            warnings.warn("No columns were selected for output.  This may be because all columns were anonymized.")
    def _fit(self, row):
        for v, t in zip(row, self.transformers):
            t._fit(v)
    def _fit_finish(self):
        self.output_width = sum([t.output_width for t in self.transformers])
    def transform(self, data):
        """Transforms the data.

        :param data: tabular data to transform
        :type data: a list of tuples, a numpy.ndarray, or a pandas DataFrame
        :returns: the transformed data
        :rtype: a list of tuples
        """
        # always returns a list of tuples, except in null case
        if self.transformers == []:
            return data
        if isinstance(data, pd.DataFrame):
            if self._columns is not None:
                # check that columns match
                columns = list(data.columns)
                if len(columns) != len(self._columns):
                    raise ValueError(f"Wrong number of columns: got {len(columns)}, expected {len(self._columns)}")
                if not(all([a == b for a, b in zip(columns, self._columns)])):
                    warnings.warn(f"Columns of data do not match columns of transformer: {columns} vs {self._columns}")
            self._columns = data.columns
            data = [tuple([c for c in t[1:]]) for t in data.itertuples()]
        elif isinstance(data, np.ndarray):
            self._dtype = data.dtype
            if len(data.shape) != 2:
                raise ValueError(f"Data must be a 2D array, got shape {data.shape}")
            if data.shape[1] != len(self.transformers):
                raise ValueError(f"Data must have {len(self.transformers)} columns, got {data.shape[1]}")
            data = [tuple([c for c in t]) for t in data]
        return [self._transform(row) for row in data]
    def _transform(self, row):
        out_row = []
        for v, t in zip(row, self.transformers):
            if isinstance(t, AnonymizationTransformer) and not t.fake_inbound:
                pass  # don't include any values if we wish to anonymize with inverse transformation
            elif isinstance(t, DropTransformer):
                pass  # don't include any values from DropTransformer
            elif t.output_width == 1:
                out_row.append(t._transform(v))
            else:
                for out_v in t._transform(v):
                    out_row.append(out_v)
        return tuple(out_row)
    def fit_transform(self, data, *ignore, epsilon=None):
        """Fits the transformer to the data, then transforms.

        :param data: tabular data to transform
        :type data: a list of tuples, a numpy.ndarray, or a pandas DataFrame
        :param epsilon: the privacy budget to spend fitting the data
        :type epsilon: float, optional
        :returns: the transformed data
        :rtype: a list of tuples
        """
        self.fit(data, epsilon=epsilon)
        return self.transform(data)
    def inverse_transform(self, data):
        if self.transformers == []:
            return data
        transformed = [self._inverse_transform(row) for row in data]
        if self._columns is not None:
            columns = [col for i, col in enumerate(self._columns) if i not in self._dropped_column_indices]
            return pd.DataFrame(transformed, columns=columns)
        elif self._dtype is not None:
            return np.array(transformed, dtype=self._dtype)
        else:
            return transformed
    def _inverse_transform(self, row):
        if len(row) != self.output_width:
            raise ValueError(f"Row has wrong length: got {len(row)}, expected {self.output_width}")
        out_row = []
        row = list(row)
        for i, t in enumerate(self.transformers):
            if isinstance(t, DropTransformer): # don't include None values from DropTransformer
                self._dropped_column_indices.add(i) # and mark column index as dropped
                continue

            if t.output_width == 1:
                v = row.pop(0)
            else:
                v = tuple([row.pop(0) for _ in range(t.output_width)])
            out_row.append(t._inverse_transform(v))
        return tuple(out_row)

    # factory methods
    @classmethod
    def from_column_names(cls, column_names, style='gan', *ignore, nullable=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[], special_types={}, constraints=None):
        transformers = TypeMap.get_transformers(column_names, style=style, nullable=nullable, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, special_types=special_types, constraints=constraints)
        return cls(transformers)
    @classmethod
    def from_pandas(cls, df, style='gan', *ignore, nullable=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[], special_types={}, constraints=None):
        return cls.from_column_names(df.columns, style=style, nullable=nullable, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, special_types=special_types, constraints=constraints)
    @classmethod
    def from_list(cls, data, style='gan', *ignore, nullable=False, header=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[], special_types={}, constraints=None):
        if not header:
            column_names = list(range(len(data[0])))
        else:
            column_names = data[0]
        return cls.from_column_names(column_names, style=style, nullable=nullable, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, special_types=special_types, constraints=constraints)
    @classmethod
    def from_numpy(cls, data, style='gan', *ignore, nullable=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[], special_types={}, constraints=None):
        column_names = list(range(len(data[0])))
        return cls.from_column_names(column_names, style=style, nullable=nullable, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, special_types=special_types, constraints=constraints)
    @classmethod
    def create(cls, data, style='gan', *ignore, nullable=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[], special_types={}, constraints=None):
        """
        Creates a transformer for the given data. Infers all columns if the provided lists are empty.
        Columns that are referenced in a constraint will be excluded from the inference.

        :param data: The private data to construct a transformer for.
        :type data: pd.DataFrame, np.ndarray, or list of tuples
        :param style: The style influences the choice of ColumnTransformers. Can either be 'gan' or 'cube'.
            Defaults to 'gan' which results in one-hot style.
        :type style: string, optional
        :param nullable: Whether to allow null values in the data. This is used as a hint when inferring transformers.
            Defaults to False.
        :type nullable: bool, optional
        :param categorical_columns: List of column names or indixes to be treated as categorical columns, used as hint.
        :type categorical_columns: list[], optional
        :param ordinal_columns: List of column names or indices to be treated as ordinal columns, used as hint.
        :type ordinal_columns: list[], optional
        :param continuous_columns: List of column names or indices to be treated as continuous columns, used as hint.
        :type continuous_columns: list[], optional
        :param constraints: Dictionary that maps from column names or indixes to constraints.
            There are multiple ways to specify a constraint.
            It can be a ``ColumnTransformer`` object, type or class name.
            Another possiblity is the string keyword 'drop' which enforces a ``DropTransformer``.
            Also, a string alias for any of the lists like 'categorical' can be provided.
            All other values e.g. a callable or Faker method will be passed into an ``AnonymizationTransformer``.
        :type constraints: dict, optional
        :returns: The transformer object
        :rtype: TableTransformer
        """
        if categorical_columns is None:
            categorical_columns = []
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []
        if special_types is None:
            special_types = {}

        excluded_columns = None
        if constraints is not None: # exclude columns with constraint from inference
            if not isinstance(constraints, dict):
                raise ValueError("Provided `constraints` is invalid. Must be a dictionary or None.")
            excluded_columns = set(constraints.keys())

        if len(continuous_columns) + len(ordinal_columns) + len(categorical_columns) == 0:
            inferred = TypeMap.infer_column_types(data, excluded_columns=excluded_columns)
            categorical_columns = inferred['categorical_columns']
            ordinal_columns = inferred['ordinal_columns']
            continuous_columns = inferred['continuous_columns']
            special_types = dict(zip(inferred['columns'], inferred['pii']))
            if not nullable:
                nullable = len(inferred['nullable_columns']) > 0
        all_specified = list(categorical_columns) + list(ordinal_columns) + list(continuous_columns)
        all_numeric = all([isinstance(c, int) for c in all_specified])
        if isinstance(data, pd.DataFrame):
            return cls.from_pandas(data, style=style, nullable=nullable, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, special_types=special_types, constraints=constraints)
        elif isinstance(data, np.ndarray):
            return cls.from_numpy(data, style=style, nullable=nullable, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, special_types=special_types, constraints=constraints)
        elif isinstance(data, list):
            return cls.from_list(data, style=style, nullable=nullable, header=(not all_numeric), categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, special_types=special_types, constraints=constraints)
        else:
            raise ValueError(f"Unknown data type: {type(data)}")

class NoTransformer(TableTransformer):
    """A pass-through table transformer that does nothing.  Note that the ``transform`` and ``inverse_transform`` methods
    will simply return the data that is passed in, rather than transforming to and from a list of tuples.  This transformer
    is suitable when you know that your input data is exactly what is needed for a specific synthesizer, and
    you want to skip all pre-processing steps.  If you want a passthrough transformer that is slightly more
    adaptable to multiple synthesizers, you can make a new ``TableTransformer`` with a list of ``IdentityTransformer``
    column transformers.
    """
    def __init__(self, *ignore):
        super().__init__([])
