from .base import ColumnTransformer

class ClampTransformer(ColumnTransformer):
    """Clamps values to be within a specified range.

    :param lower: The minimum value to scale to.  If None, no lower bound is applied.
    :param upper: The maximum value to scale to.  If None, no upper bound is applied.
    """
    def __init__(self, upper=None, lower=None):
        super().__init__()
        if upper is None and lower is None:
            raise ValueError('Must specify upper and/or lower')
        self.upper = upper
        self.lower = lower
    @property
    def cardinality(self):
        return [1]
    def _fit(self, val):
        pass
    def _clear_fit(self):
        self._fit_complete = True
        self.output_width = 1
    def _transform(self, val):
        if self.upper is not None and val > self.upper:
            return self.upper
        if self.lower is not None and val < self.lower:
            return self.lower
        return val
    def _inverse_transform(self, val):
        if self.upper is not None and val > self.upper:
            return self.upper
        if self.lower is not None and val < self.lower:
            return self.lower
        return val
