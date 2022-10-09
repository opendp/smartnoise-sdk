from snsynth.transform.definitions import ColumnType
from .base import ColumnTransformer
import numpy as np

class OneHotEncoder(ColumnTransformer):
    """Transforms integer-labeled data into one-hot encoding.  Inputs are assumed to be 0-based.
    To convert from unstructured categorical data, chain with LabelTransformer first.
    """
    cache_fit = False
    def __init__(self):
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CATEGORICAL
    @property
    def cardinality(self):
        return [2] * (self.max + 1)
    def _fit(self, val):
        if val > self.max:
            self.max = val
    def _fit_finish(self):
        self.output_width = self.max + 1
        super()._fit_finish()
    def _clear_fit(self):
        self._fit_complete = False
        self.max = -1
    def _transform(self, val):
        if self.max < 0 or not self._fit_complete:
            raise ValueError("OneHotEncoder has not been fit yet.")
        bits = [0] * (self.max + 1)
        bits[val] = 1
        return tuple(bits)
    def _inverse_transform(self, val):
        # will always choose first if multiple are set
        return np.argmax(val)
