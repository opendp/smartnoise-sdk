from .base import CachingColumnTransformer

class ChainTransformer(CachingColumnTransformer):
    def __init__(self, transformers):
        self.transformers = transformers
        super().__init__()
    @property
    def fit_complete(self):
        return all([t.fit_complete for t in self.transformers])
    def _fit_finish(self):
        vals = self._fit_vals
        for transformer in self.transformers:
            vals = transformer.fit_transform(vals)
        self._fit_vals = []
        self.output_width = self.transformers[-1].output_width
    def _clear_fit(self):
        for transformer in self.transformers:
            transformer._clear_fit()
        if self.fit_complete:
            self.output_width = self.transformers[-1].output_width
    def _transform(self, val):
        for transformer in self.transformers:
            val = transformer._transform(val)
        return val
    def _inverse_transform(self, val):
        for transformer in reversed(self.transformers):
            val = transformer._inverse_transform(val)
        return val