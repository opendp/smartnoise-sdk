from snsynth.transform.minmax import MinMaxTransformer
from snsynth.transform.bin import BinTransformer
from snsynth.transform.label import LabelTransformer
from snsynth.transform.onehot import OneHotEncoder

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