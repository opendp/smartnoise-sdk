import pandas as pd
import numpy as np
import warnings
from snsql.sql.odometer import OdometerHeterogeneous
from snsql.sql.privacy import Privacy

from snsynth.transform.bin import BinTransformer
from snsynth.transform.chain import ChainTransformer
from snsynth.transform.label import LabelTransformer
from snsynth.transform.minmax import MinMaxTransformer
from snsynth.transform.onehot import OneHotEncoder

class TableTransformer:
    def __init__(self, transformers, *ignore, odometer=None):
        self._columns = None
        # one transformer per column
        self.transformers = transformers
        if self.fit_complete:
            self.output_width = sum([t.output_width for t in self.transformers])
        else:
            self.output_width = 0
        if odometer is None:
            self.odometer = OdometerHeterogeneous(privacy=Privacy())
        else:
            self.odometer = odometer
    @property
    def fit_complete(self):
        return all([t.fit_complete for t in self.transformers])
    def allocate_privacy_budget(self, epsilon, odometer):
        n_with_epsilon = sum([1 for t in self.transformers if t.needs_epsilon])
        if n_with_epsilon == 0:
            return
        else:
            for transformer in self.transformers:
                if transformer.needs_epsilon:
                    transformer.allocate_privacy_budget(epsilon / n_with_epsilon, odometer)
    def fit(self, data, *ignore, epsilon=None):
        if epsilon is not None and epsilon > 0.0:
            self.allocate_privacy_budget(epsilon, self.odometer)
        if isinstance(data, pd.DataFrame):
            self._columns = list(data.columns)
            data = [tuple([c for c in t[1:]]) for t in data.itertuples()]
        for t in self.transformers:
            t._clear_fit()
        for row in data:
            self._fit(row)
        for t in self.transformers:
            t._fit_finish()
        self._fit_finish()
    def _fit(self, row):
        for v, t in zip(row, self.transformers):
            t._fit(v)
    def _fit_finish(self):
        self.output_width = sum([t.output_width for t in self.transformers])
    def transform(self, data):
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
            if len(data.shape) != 2:
                raise ValueError(f"Data must be a 2D array, got shape {data.shape}")
            if data.shape[1] != len(self.transformers):
                raise ValueError(f"Data must have {len(self.transformers)} columns, got {data.shape[1]}")
            data = [tuple([c for c in t]) for t in data]
        return [self._transform(row) for row in data]
    def _transform(self, row):
        out_row = []
        for v, t in zip(row, self.transformers):
            if t.output_width == 1:
                out_row.append(t._transform(v))
            else:
                for out_v in t._transform(v):
                    out_row.append(out_v)
        return tuple(out_row)
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    def inverse_transform(self, data):
        transformed = [self._inverse_transform(row) for row in data]
        if self._columns is not None:
            return pd.DataFrame(transformed, columns=self._columns)
        else:
            return transformed
    def _inverse_transform(self, row):
        if len(row) != self.output_width:
            raise ValueError(f"Row has wrong length: got {len(row)}, expected {self.output_width}")
        out_row = []
        row = list(row)
        for t in self.transformers:
            if t.output_width == 1:
                v = row.pop(0)
            else:
                v = tuple([row.pop(0) for _ in range(t.output_width)])
            out_row.append(t._inverse_transform(v))
        return tuple(out_row)
    @classmethod
    def from_column_names(cls, column_names, style='gan', *ignore, categorical_columns=[], ordinal_columns=[], continuous_columns=[]):
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []
        if categorical_columns is None:
            categorical_columns = []
        transformers = []
        for col in list(column_names):
            if col in categorical_columns:
                if style == 'gan':
                    t = ChainTransformer([LabelTransformer(), OneHotEncoder()])
                    transformers.append(t)
                elif style == 'cube':
                    t = LabelTransformer()
                    transformers.append(t)
                else:
                    raise ValueError(f"Unknown style: {style}")
            elif col in ordinal_columns:
                if style == 'gan':
                    t = ChainTransformer([LabelTransformer(), OneHotEncoder()])
                    transformers.append(t)
                elif style == 'cube':
                    t = LabelTransformer()
                    transformers.append(t)
                else:
                    raise ValueError(f"Unknown style: {style}")
            elif col in continuous_columns:
                if style == 'gan':
                    t = MinMaxTransformer(epsilon=0.0, nullable=True)
                    transformers.append(t)
                elif style == 'cube':
                    t = BinTransformer(epsilon=0.0, nullable=True)
                    transformers.append(t)
                else:
                    raise ValueError(f"Unknown style: {style}")
            else:
                raise ValueError(f"Column in dataframe not specified as categorical, ordinal, or continuous: {col}")
        return cls(transformers)
    @classmethod
    def from_pandas(cls, df, style='gan', *ignore, categorical_columns=[], ordinal_columns=[], continuous_columns=[]):
        return cls.from_column_names(df.columns, style=style, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns)
    @classmethod
    def from_list(cls, data, style='gan', *ignore, header=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[]):
        if not header:
            column_names = list(range(len(data[0])))
        else:
            column_names = data[0]
        return cls.from_column_names(column_names, style=style, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns)
    @classmethod
    def from_numpy(cls, data, style='gan', *ignore, categorical_columns=[], ordinal_columns=[], continuous_columns=[]):
        column_names = list(range(len(data[0])))
        return cls.from_column_names(column_names, style=style, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns)
    @classmethod
    def create(cls, data, style='gan', *ignore, categorical_columns=[], ordinal_columns=[], continuous_columns=[]):
        if categorical_columns is None:
            categorical_columns = []
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []
        all_specified = list(categorical_columns) + list(ordinal_columns) + list(continuous_columns)
        all_numeric = all([isinstance(c, int) for c in all_specified])
        if isinstance(data, pd.DataFrame):
            return cls.from_pandas(data, style=style, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns)
        elif isinstance(data, np.ndarray):
            return cls.from_numpy(data, style=style, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns)
        elif isinstance(data, list):
            return cls.from_list(data, style=style, header=(not all_numeric), categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns)
        else:
            raise ValueError(f"Unknown data type: {type(data)}")

