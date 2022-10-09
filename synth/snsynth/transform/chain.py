from .base import CachingColumnTransformer
import warnings

class ChainTransformer(CachingColumnTransformer):
    """Sequentially process a column through multiple transforms.  When reversed,
    the inverse transforms are applied in reverse order.

    :param transforms: A list of ColumnTransformers to apply sequentially.
    """
    def __init__(self, transformers):
        self.transformers = transformers
        super().__init__()
    @property
    def output_type(self):
        return self.transformers[-1].output_type
    @property
    def needs_epsilon(self):
        return any(transformer.needs_epsilon for transformer in self.transformers)
    @property
    def cardinality(self):
        cards = []
        for transformer in self.transformers:
            for c in transformer.cardinality:
                cards.append(c)
        return cards
    @property
    def fit_complete(self):
        return all([t.fit_complete for t in self.transformers])
    def allocate_privacy_budget(self, epsilon, odometer):
        n_with_epsilon = sum([1 for t in self.transformers if t.needs_epsilon])
        if n_with_epsilon == 0:
            return
        elif n_with_epsilon > 1:
            warnings.warn(f"Multiple transformers in chain need epsilon, which is likely wasteful.")
        else:
            for transformer in self.transformers:
                if transformer.needs_epsilon:
                    transformer.allocate_privacy_budget(epsilon / n_with_epsilon, odometer)
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