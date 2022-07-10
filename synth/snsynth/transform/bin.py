from .base import CachingColumnTransformer
from .mechanism import approx_bounds

class BinTransformer(CachingColumnTransformer):
    """Transformer that bins values into a discrete set of bins.

    :param lower: The minimum value to scale to.
    :param upper: The maximum value to scale to.
    :param bins: The number of bins to use.
    :param negative: Whether to scale between -1.0 and 1.0.
    :param epsilon: The privacy budget to use.
    :return: A transformed column of values.
    """
    def __init__(self, *, bins=10, lower=None, upper=None, epsilon=None):
        if epsilon is None and (lower is None and upper is None):
            raise ValueError("BinTransformer requires either epsilon or upper and lower.")
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon
        self.bins = bins
        self.budget_spent = []
        super().__init__()
    def _fit_finish(self):
        if self.epsilon is not None and (self.lower is None or self.upper is None):
            self.fit_lower, self.fit_upper = approx_bounds(self._fit_vals, self.epsilon)
            self.budget_spent.append(self.epsilon)
            if self.fit_lower is None or self.fit_upper is None:
                raise ValueError("BinTransformer could not find bounds.")
        elif self.lower is None or self.upper is None:
            raise ValueError("BinTransformer requires either epsilon or min and max.")
        else:
            self.fit_lower = self.lower
            self.fit_upper = self.upper
        self._fit_complete = True
        self.output_width = 1
    def _clear_fit(self):
        self._reset_fit()
        self.fit_lower = None
        self.fit_upper = None
        # if bounds provided, we can immediately use without fitting
        if self.lower and self.upper:
            self._fit_complete = True
            self.output_width = 1
            self.fit_lower = self.lower
            self.fit_upper = self.upper
    def _bin_edges(self, bin):
        return (
            self.fit_lower + (bin / self.bins) * (self.fit_upper - self.fit_lower),
            self.fit_lower + ((bin + 1) / self.bins) * (self.fit_upper - self.fit_lower)
        )
    def _bin(self, val):
        if not self.fit_complete:
            raise ValueError("BinTransformer has not been fit yet.")
        return int(self.bins * (val - self.fit_lower) / (self.fit_upper - self.fit_lower))
    def _transform(self, val):
        if not self.fit_complete:
            raise ValueError("BinTransformer has not been fit yet.")
        val = self.fit_lower if val < self.fit_lower else val
        val = self.fit_upper if val > self.fit_upper else val
        return self._bin(val)
    def _inverse_transform(self, val):
        if not self.fit_complete:
            raise ValueError("BinTransformer has not been fit yet.")
        lower, upper = self._bin_edges(val)
        return (lower + upper) / 2
