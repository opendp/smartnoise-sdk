from snsynth.transform.definitions import ColumnType
from .base import ColumnTransformer
import numpy as np

class LabelTransformer(ColumnTransformer):
    def __init__(self):
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CATEGORICAL
    def _fit(self, val):
        if isinstance(val, float) and np.isnan(val):
            val = None
        if val not in self.labels:
            self.labels[val] = self.category
            self.categories[self.category] = val
            self.category += 1
            self.output_width = 1
    def _clear_fit(self):
        self._reset_fit()
        self.labels = {}
        self.categories = {}
        self.category = 0
    def _transform(self, val):
        if isinstance(val, float) and np.isnan(val):
            val = None
        return self.labels[val]
    def _inverse_transform(self, val):
        return self.categories[val]
