import pandas as pd
import numpy as np
import warnings
import re
from snsynth import transform
from snsynth.transform.anonymization import AnonymizationTransformer
from snsynth.transform.base import ColumnTransformer
from snsynth.transform.drop import DropTransformer
from snsynth.transform.minmax import MinMaxTransformer
from snsynth.transform.bin import BinTransformer
from snsynth.transform.label import LabelTransformer
from snsynth.transform.onehot import OneHotEncoder
from snsynth.transform.chain import ChainTransformer
from snsynth.transform.datetime import DateTimeTransformer

class SequenceCounter:
    def __init__(self):
        self.count = -1
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.count

_EXPECTED_COL_STYLES = {
    'gan': {
        'categorical': [OneHotEncoder],
        'ordinal': [OneHotEncoder],
        'continuous': [MinMaxTransformer],
    },
    'cube': {
        'categorical': [LabelTransformer],
        'ordinal': [LabelTransformer],
        'continuous': [BinTransformer],
    }
}

class TypeMap:
    def __init__(self):
        pass    
    @classmethod
    def get_transformers(cls, column_names, style='gan', *ignore, nullable=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[], special_types={}, constraints=None):
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []
        if categorical_columns is None:
            categorical_columns = []
        if constraints is None:
            constraints = dict()

        constrained_columns = set(constraints.keys())
        for k, v in constraints.items(): # check if a string value is an alias for any of the lists
            if isinstance(v, str) and v.lower() in {"categorical", "ordinal", "continuous"}:
                if v.lower() == "categorical":
                  categorical_columns.append(k)
                elif v.lower() == "ordinal":
                   ordinal_columns.append(k)
                elif v.lower() == "continuous":
                    continuous_columns.append(k)
                constrained_columns.remove(k)

        transformers = []
        for col in list(column_names):
            if col in constrained_columns:
                transformer = cls._get_transformer(constraints[col])
                transformers.append(transformer)
            elif col in special_types and special_types[col] is not None:
                stype = special_types[col]
                if stype in ['email', 'ssn', 'uuid4']:
                    transformers.append(AnonymizationTransformer(stype))
                elif stype == 'sequence':
                    transformers.append(AnonymizationTransformer(SequenceCounter()))
                elif stype == 'datetime':
                    if style == 'gan':
                        t = ChainTransformer([
                            DateTimeTransformer(),
                            MinMaxTransformer(nullable=nullable)
                        ])
                        transformers.append(t)
                    elif style == 'cube':
                        t = ChainTransformer([
                            DateTimeTransformer(),
                            BinTransformer(bins=20, nullable=nullable)
                        ])
                        transformers.append(t)
                    else:
                        raise ValueError(f"Unknown style: {style}")
                else:
                    raise ValueError(f"Unknown special type {stype}")
            elif col in categorical_columns:
                if style == 'gan':
                    t = ChainTransformer([LabelTransformer(nullable=nullable), OneHotEncoder()])
                    transformers.append(t)
                elif style == 'cube':
                    t = LabelTransformer(nullable=nullable)
                    transformers.append(t)
                else:
                    raise ValueError(f"Unknown style: {style}")
            elif col in ordinal_columns:
                if style == 'gan':
                    t = ChainTransformer([LabelTransformer(nullable=nullable), OneHotEncoder()])
                    transformers.append(t)
                elif style == 'cube':
                    t = LabelTransformer(nullable=nullable)
                    transformers.append(t)
                else:
                    raise ValueError(f"Unknown style: {style}")
            elif col in continuous_columns:
                if style == 'gan':
                    t = MinMaxTransformer(nullable=nullable)
                    transformers.append(t)
                elif style == 'cube':
                    t = BinTransformer(nullable=nullable)
                    transformers.append(t)
                else:
                    raise ValueError(f"Unknown style: {style}")
            else:
                raise ValueError(f"Column in dataframe not specified as categorical, ordinal, continuous or constraint: {col}")
        return transformers
    @classmethod
    def infer_column_types(cls, data, excluded_columns=None):
        max_cached = 1000
        n_columns = 0
        colnames = []
        coltypes = []
        nullable = []
        pii = []
    
        if isinstance(data, pd.DataFrame):
            colnames = list(data.columns)
            n_columns = len(colnames)
            data = [tuple([c for c in t[1:]]) for t in data.itertuples()]
        elif isinstance(data, list):
            colnames = [v for v in data[0]]
            colname_types = set([type(v) for v in colnames])
            if len(colname_types) != 1 or str not in colname_types:
                colnames = [i for i in range(len(colnames))]
            n_columns = len(colnames)
        elif isinstance(data, np.ndarray):
            n_columns = data.shape[1]
            colnames = [i for i in range(n_columns)]
            data = data.tolist()

        excluded_indices = set()
        if excluded_columns is not None and len(excluded_columns) > 0: # don't infer the excluded columns
            colnames_actual = []
            for i, c in enumerate(colnames):
                if c in excluded_columns:
                    excluded_indices.add(i)
                else:
                    colnames_actual.append(c)
            colnames = colnames_actual
            n_columns = len(colnames)

        # cache up to max_cached rows
        n_cached = 0
        value_cache = []
        for _ in range(n_columns):
            value_cache.append([])
            pii.append(None)
            nullable.append(False)
            coltypes.append(None)
        for row in data:
            index = 0
            for i, val in enumerate(row):
                if i not in excluded_indices:
                    value_cache[index].append(val)
                    index += 1
            n_cached += 1
            if n_cached >= max_cached:
                break

        # infer each column type
        for i in range(n_columns):
            # check for nulls
            if any([v is None or isinstance(v, float) and np.isnan(v) for v in value_cache[i]]):
                nullable[i] = True
            # remove nulls
            value_cache[i] = [v for v in value_cache[i] if v is not None and not (isinstance(v, float) and np.isnan(v))]
            # check for bools
            if any([isinstance(v, bool) for v in value_cache[i]]):
                coltypes[i] = 'categorical'
            # check for strings
            elif any([isinstance(v, str) for v in value_cache[i]]):
                # check for pii
                pii_type = cls.infer_pii(value_cache[i])
                if pii_type is not None:
                    pii[i] = pii_type
                else:
                    distinct = set(value_cache[i])
                    if len(distinct) > 0.8 * len(value_cache[i]):
                        warnings.warn(f"Column {colnames[i]} has {len(distinct)} distinct values out of {len(value_cache[i])} total values. This appears to be unbounded categorical data and may risk privacy leaks.")
                coltypes[i] = 'categorical'
            elif any([isinstance(v, int) for v in value_cache[i]]):
                distinct = set(value_cache[i])
                if len(distinct) < 150 and max(distinct) - min(distinct) < 150:
                    coltypes[i] ='ordinal'
                else:
                    # check for sequence
                    if cls.is_ids(value_cache[i]):
                        pii[i] = 'sequence'
                    coltypes[i] = 'continuous'
            elif any([isinstance(v, float) for v in value_cache[i]]):
                if all([v.is_integer() for v in value_cache[i]]):
                    distinct = set(value_cache[i])
                    threshold = n_cached * 0.20 # 20% of rows
                    if len(distinct) < threshold and max(distinct) - min(distinct) < threshold:
                        coltypes[i] = 'ordinal'
                    else:
                        # check for sequence
                        if cls.is_ids(value_cache[i]):
                            pii[i] = 'sequence'
                        coltypes[i] = 'continuous'
                else:
                    coltypes[i] = 'continuous'
            else:
                distinct = set(value_cache[i])
                threshold = n_cached * 0.20 # 20% of rows
                if len(distinct) < threshold:
                    coltypes[i] = 'categorical'
                else:
                    raise ValueError(f"Cannot infer a column type for column {i}")
        
        result = {
            'columns': colnames,
            'categorical_columns': [colnames[i] for i, v in enumerate(coltypes) if v == 'categorical'],
            'ordinal_columns': [colnames[i] for i, v in enumerate(coltypes) if v == 'ordinal'],
            'continuous_columns': [colnames[i] for i, v in enumerate(coltypes) if v == 'continuous'],
            'nullable_columns': [colnames[i] for i, v in enumerate(nullable) if v],
            'pii': pii
        }
        return result

    @classmethod
    def infer_pii(cls, vals):
        uuid4string = re.compile(r'[0-9a-f]{8}-?[0-9a-f]{4}-?4[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}')
        email = re.compile(r'[^@]+@[^@]+\.[^@]+')
        ssn = re.compile(r'^\d{3}-?\d{2}-?\d{4}$')
        dt = DateTimeTransformer()
        types = []
        for v in vals:
            try:
                parsed = dt._parse_date(v)
                if parsed is not None:
                    types.append('datetime')
            except:
                pass
            if uuid4string.match(v):
                types.append('uuid4')
            elif email.match(v):
                types.append('email')
            elif ssn.match(v):
                types.append('ssn')
        if len(types) == 0:
            return None
        types = list(set(types))
        if len(types) == 1:
            return types[0]
        else:
            warnings.warn(f"Multiple PII types detected: {types}")
            return types[0]

    @classmethod
    def is_ids(cls, vals):
        # must pass in ints or floats
        distinct = set(vals)
        if len(distinct) < 0.8 * len(vals):
            return False
        _range = max(distinct) - min(distinct)
        if _range >= len(distinct) and _range < 1.1 * len(distinct):
            return True
        vals = sorted(list(distinct))
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        if len(set(diffs)) < 4:
            return True
        if np.std(diffs) < 0.1 * np.mean(diffs):
            return True
        return False

    @staticmethod
    def _get_transformer(constraint):
        """
        Constructs a transformer object according to the given constraint.
        Firstly, if the string is a keyword e.g. 'drop' the corresponding transformer is returned.
        Secondly, the method tries to construct a transformer class from a string value.
        Other values will be passed into an ``AnonymizationTransformer``.

        :param constraint: Transformer to be used, a keyword / built-in Faker method
        :type constraint: a ColumnTransformer object/type, a string, or a callable
        :return: Transformer object
        :rtype: ColumnTransformer
        """
        if isinstance(constraint, ColumnTransformer):
            return constraint
        elif isinstance(constraint, type) and issubclass(constraint, ColumnTransformer):
            return constraint()
        elif isinstance(constraint, str):
            # firstly, check if column should be ignored completely
            if constraint.lower() == "drop":
                return DropTransformer()

            # secondly, try to find a valid transformer class
            try:
                transformer_class = getattr(transform, constraint)
                return transformer_class()
            except AttributeError:
                pass

        # otherwise assume that e.g. a Faker method is given
        try:
            return AnonymizationTransformer(constraint)
        except ValueError as e:
            raise ValueError(
                f"Provided `constraint` {constraint} is invalid. Please read the documentation about constraints."
            ) from e
