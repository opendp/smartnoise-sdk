from .base import CachingColumnTransformer
from snsql.sql._mechanisms.approx_bounds import approx_bounds
from snsql.sql.privacy import Privacy
from snsynth.transform.definitions import ColumnType
import numpy as np

class BinTransformer(CachingColumnTransformer):
    """Transforms continuous values into a discrete set of bins.

    :param bins: The number of bins to create.
    :param lower: The minimum value to scale to.
    :param upper: The maximum value to scale to.
    :param epsilon: The privacy budget to use to infer bounds, if none provided.
    :param nullable: If null values are expected, a second output will be generated indicating null.
    :param odometer: The optional odometer to use to track privacy budget.
    """
    def __init__(self, *, bins=10, lower=None, upper=None, epsilon=0.0, nullable=False, odometer=None):
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon
        self.bins = bins
        self.budget_spent = []
        self.nullable = nullable
        self.odometer = odometer
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CATEGORICAL
    @property
    def needs_epsilon(self):
        return self.upper is None or self.lower is None
    @property
    def cardinality(self):
        if self.nullable:
            return [self.bins, 2]
        else:
            return [self.bins]
    def allocate_privacy_budget(self, epsilon, odometer):
        self.epsilon = epsilon
        self.odometer = odometer
    def _fit_finish(self):
        if self.epsilon is not None and self.epsilon > 0.0 and (self.lower is None or self.upper is None):
            self._fit_vals = [v for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            self.fit_lower, self.fit_upper = approx_bounds(self._fit_vals, self.epsilon)
            if self.odometer is not None:
                self.odometer.spend(Privacy(epsilon=self.epsilon, delta=0.0))
            self.budget_spent.append(self.epsilon)
            if self.fit_lower is None or self.fit_upper is None:
                raise ValueError("BinTransformer could not find bounds.")
        elif self.lower is None or self.upper is None:
            raise ValueError("BinTransformer requires either epsilon or min and max.")
        else:
            self.fit_lower = self.lower
            self.fit_upper = self.upper
        self._fit_complete = True
        if self.nullable:
            self.output_width = 2
        else:
            self.output_width = 1
    def _clear_fit(self):
        self._reset_fit()
        self.fit_lower = None
        self.fit_upper = None
        # if bounds provided, we can immediately use without fitting
        if self.lower and self.upper:
            self._fit_complete = True
            if self.nullable:
                self.output_width = 2
            else:
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
        if self.nullable and (val is None or (isinstance(val, float) and np.isnan(val))):
            return 1
        return int(self.bins * (val - self.fit_lower) / (self.fit_upper - self.fit_lower))
    def _transform(self, val):
        if not self.fit_complete:
            raise ValueError("BinTransformer has not been fit yet.")
        if  val is None or (isinstance(val, float) and np.isnan(val)):
            if self.nullable:
                return (0, 1)
            else:
                raise ValueError("Cannot transform None or NaN.  Consider setting nullable=True.")
        val = self.fit_lower if val < self.fit_lower else val
        val = self.fit_upper if val > self.fit_upper else val
        if self.nullable:
            return (self._bin(val), 0)
        else:
            return self._bin(val)
    def _inverse_transform(self, val):
        if not self.fit_complete:
            raise ValueError("BinTransformer has not been fit yet.")
        if self.nullable:
            v, n = val
            if n == 1 or v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            val = v
        lower, upper = self._bin_edges(val)
        return (lower + upper) / 2
