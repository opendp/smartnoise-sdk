from snsynth.transform.definitions import ColumnType
from .base import ColumnTransformer
import numpy as np

class LogTransformer(ColumnTransformer):
    """Logarithmic transformation of values.  Useful for transforming skewed data.

    """
    def __init__(self):
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CONTINUOUS
    @property
    def cardinality(self):
        return [None]
    def _fit(self, val, idx=None):
        pass
    def _clear_fit(self):
        # this transform doesn't need fit
        self._fit_complete = True
        self.output_width = 1
    def _transform(self, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        else:
            return float(np.log(val))
    def _inverse_transform(self, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        else:
            return float(np.exp(val))