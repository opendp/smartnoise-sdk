from snsynth.transform.definitions import ColumnType
from .base import ColumnTransformer
import numpy as np

class LogTransformer(ColumnTransformer):
    def __init__(self):
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CONTINUOUS
    def _fit(self, val, idx=None):
        pass
    def _clear_fit(self):
        # this transform doesn't need fit
        self._fit_complete = True
        self.output_width = 1
    def _transform(self, val):
        return float(np.log(val))
    def _inverse_transform(self, val):
        return float(np.exp(val))