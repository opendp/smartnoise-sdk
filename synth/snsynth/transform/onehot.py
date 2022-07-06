from .base import ColumnTransformer

class OneHotEncoder(ColumnTransformer):
    cache_fit = False
    def __init__(self):
        super().__init__()
        self.max = -1
    def _fit(self, val):
        if val > self.max:
            self.max = val
    def _transform(self, val):
        if self.max < 0 or not self._fit_complete:
            raise ValueError("OneHotEncoder has not been fit yet.")
        bits = [0] * (self.max + 1)
        bits[val] = 1
        return tuple(bits)
    def _inverse_transform(self, val):
        if 1 not in val:
            raise ValueError("OneHotEncoder attempting to invert transform with no bits set.")
        return val.index(1)
