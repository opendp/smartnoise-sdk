from .base import ColumnTransformer

class ChainTransformer(ColumnTransformer):
    def __init__(self, transformers):
        super().__init__()
        self.transformers = transformers
    def _fit(self, val):
        super()._fit(val)
        # for transformer in self.transformers:
        #     transformer._fit(val)
        #     val = transformer._transform(val)  # this seems wasteful
    def _fit_finish(self):
        vals = self._fit_vals
        for transformer in self.transformers:
            vals = transformer.fit_transform(vals)
        self._fit_vals = []
        self._fit_complete = True
    def _transform(self, val):
        for transformer in self.transformers:
            val = transformer._transform(val)
        return val
    def _inverse_transform(self, val):
        for transformer in reversed(self.transformers):
            val = transformer._inverse_transform(val)
        return val