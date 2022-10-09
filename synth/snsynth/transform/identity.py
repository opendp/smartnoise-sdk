from .base import ColumnTransformer

class IdentityTransformer(ColumnTransformer):
    """Dummy transformer that passes through values.  Useful for testing.
    """
    def __init__(self):
        super().__init__()
    @property
    def cardinality(self):
        return [None]
    def _fit(self, val):
        pass
    def _clear_fit(self):
        self._fit_complete = True
        self.output_width = 1
    def _transform(self, val):
        return val
    def _inverse_transform(self, val):
        return val
