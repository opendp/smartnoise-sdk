from .base import ColumnTransformer

class IdentityTransformer(ColumnTransformer):
    def __init__(self):
        super().__init__()
    def _fit(self, val):
        pass
    def _clear_fit(self):
        self._fit_complete = True
        self.output_width = 1
    def _transform(self, val):
        return val
    def _inverse_transform(self, val):
        return val
