import pandas as pd
import numpy as np
from snsynth.transform.minmax import MinMaxTransformer
from snsynth.transform.bin import BinTransformer
from snsynth.transform.label import LabelTransformer
from snsynth.transform.onehot import OneHotEncoder
from snsynth.transform.chain import ChainTransformer

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
    @classmethod
    
    @classmethod
    def get_transformers(cls, column_names, style='gan', *ignore, nullable=False, categorical_columns=[], ordinal_columns=[], continuous_columns=[]):
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
                raise ValueError(f"Column in dataframe not specified as categorical, ordinal, or continuous: {col}")
        return transformers
    @classmethod
    def infer_column_types(cls, data):
        n_columns = 0
        colnames = []
        coltypes = []
        nullable = []
    
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

        n_cached = 0
        value_cache = []
        for _ in range(n_columns):
            value_cache.append([])
        for row in data:
            for i, val in enumerate(row):
                value_cache[i].append(val)
            n_cached += 1
            if n_cached >= 1000:
                break
        for i in range(n_columns):
            if any([v is None or isinstance(v, float) and np.isnan(v) for v in value_cache[i]]):
                nullable.append(True)
            else:
                nullable.append(False)
            value_cache[i] = [v for v in value_cache[i] if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if any([isinstance(v, str) or isinstance(v, bool) for v in value_cache[i]]):
                coltypes.append('categorical')
            elif any([isinstance(v, int) for v in value_cache[i]]):
                distinct = set(value_cache[i])
                if len(distinct) < 150 and max(distinct) - min(distinct) < 150:
                    coltypes.append('ordinal')
                else:
                    coltypes.append('continuous')
            elif any([isinstance(v, float) for v in value_cache[i]]):
                if all([v.is_integer() for v in value_cache[i]]):
                    distinct = set(value_cache[i])
                    if len(distinct) < 150 and max(distinct) - min(distinct) < 150:
                        coltypes.append('ordinal')
                    else:
                        coltypes.append('continuous')
                else:
                    coltypes.append('continuous')
            else:
                v = set(value_cache[i])
                if len(v) < 150:
                    coltypes.append('categorical')
                else:
                    raise ValueError(f"Cannot infer a column type for column {i}")
        
        result = {
            'columns': colnames,
            'categorical_columns': [colnames[i] for i, v in enumerate(coltypes) if v == 'categorical'],
            'ordinal_columns': [colnames[i] for i, v in enumerate(coltypes) if v == 'ordinal'],
            'continuous_columns': [colnames[i] for i, v in enumerate(coltypes) if v == 'continuous'],
            'nullable_columns': [colnames[i] for i, v in enumerate(nullable) if v]
        }
        return result