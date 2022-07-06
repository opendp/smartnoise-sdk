from .base import ColumnTransformer

class MinMaxTransformer(ColumnTransformer):
    def __init__(self, *, min=None, max=None, negative=False):
        super().__init__()
        self.min = min
        self.max = max
        self.negative = negative
        if self.min is not None and self.max is not None:
            self.fit = True
    def _fit(self, val):
        if not self.fit:
            super()._fit(val)
    def _fit_finish(self):
        self.min = min(self._fit_vals)
        self.max = max(self._fit_vals)
        self._fit_vals = []
        self._fit_complete = True
    def _transform(self, val):
        if not (self.min and self.max):
            raise ValueError("MinMaxTransformer has not been fit yet.")
        val = self.min if val < self.min else val
        val = self.max if val > self.max else val
        val = (val - self.min) / (self.max - self.min)
        if self.negative:
            val = (val * 2) - 1
        return val
    def _inverse_transform(self, val):
        if not self.min and self.max:
            raise ValueError("MinMaxTransformer has not been fit yet.")
        if self.negative:
            val = (1 + val) / 2
        val = val * (self.max - self.min) + self.min
        return val
