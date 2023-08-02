from snsynth.transform.definitions import ColumnType
from .base import CachingColumnTransformer
from opendp.mod import enable_features, binary_search_param
from opendp.transformations import make_sized_bounded_mean, make_sized_bounded_variance, make_clamp, make_resize
from opendp.domains import atom_domain
from opendp.measurements import make_base_laplace
from snsql.sql._mechanisms.approx_bounds import approx_bounds
from snsql.sql.privacy import Privacy
import numpy as np

class StandardScaler(CachingColumnTransformer):
    """Transforms a column of values to scale with mean centered on 0 and unit variance.
    Some privacy budget is always used to estimate the mean and variance.  If upper and lower
    are not supplied, the budget will also be used to estimate the bounds of the column.

    :param lower: The minimum value to scale to.
    :param upper: The maximum value to scale to.
    :param epsilon: The privacy budget to use.
    :param nullable: Whether the column can contain null values.  If True, the output will be a tuple of (value, null_flag).
    :param odometer: The optional privacy odometer to use to track privacy budget spent.
    """
    def __init__(self, *, lower=None, upper=None, epsilon=0.0, nullable=False, odometer=None):
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon
        self.budget_spent = []
        self.nullable = nullable
        self.odometer = odometer
        self.scaler = None
        self.mean = None
        self.var = None
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CONTINUOUS
    @property
    def needs_epsilon(self):
        return True
    @property
    def cardinality(self):
        if self.nullable:
            return [None, 2]
        else:
            return [None]
    def allocate_privacy_budget(self, epsilon, odometer):
        self.epsilon = epsilon
        self.odometer = odometer
    def _fit_finish(self):
        if self.scaler is None:
            if self.epsilon is None or self.epsilon == 0.0:
                raise ValueError("StandardScaler requires epsilon to estimate mean and variance.")
            self._fit_vals = [float(v) for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            # set bounds
            if self.upper is None or self.lower is None:
                bounds_eps = self.epsilon / 2
                self.epsilon -= bounds_eps
                self.fit_lower, self.fit_upper = approx_bounds(self._fit_vals, bounds_eps)
                if self.odometer is not None:
                    self.odometer.spend(Privacy(epsilon=bounds_eps, delta=0.0))
                self.budget_spent.append(bounds_eps)
                if self.fit_lower is None or self.fit_upper is None:
                    raise ValueError("StandardScaler could not find upper and lower bounds.")
            else:
                self.fit_lower = self.lower
                self.fit_upper = self.upper
            # fit scaler
            bounds = (float(self.fit_lower), float(self.fit_upper))
            n = len(self._fit_vals)

            enable_features("floating-point", "contrib")

            var_pre =  make_clamp(bounds) >> make_resize(n, atom_domain(bounds), float(self.fit_lower)) >> make_sized_bounded_variance(size=n, bounds=bounds)
            mean_pre =  make_clamp(bounds) >> make_resize(n, atom_domain(bounds), float(self.fit_lower)) >> make_sized_bounded_mean(size=n, bounds=bounds)

            v_e = self.epsilon * 0.8
            m_e = self.epsilon - v_e
            v_s = binary_search_param(lambda s: var_pre >> make_base_laplace(s), d_in=1, d_out=v_e)
            m_s = binary_search_param(lambda s: mean_pre >> make_base_laplace(s), d_in=1, d_out=m_e)
            dpvar = var_pre >> make_base_laplace(v_s)
            dpmean = mean_pre >> make_base_laplace(m_s)
            
            self.var = dpvar(np.array(self._fit_vals))
            self.var = np.clip(self.var, 0.001, (self.fit_upper - self.fit_lower) ** 2 / 4)
            self.mean = dpmean(np.array(self._fit_vals))
            self.mean = np.clip(self.mean, self.fit_lower, self.fit_upper)

            if self.odometer is not None:
                self.odometer.spend(Privacy(epsilon=self.epsilon, delta=0.0))
            self.budget_spent.append(self.epsilon)
        self._fit_complete = True
        if self.nullable:
            self.output_width = 2
        else:
            self.output_width = 1
    def _clear_fit(self):
        self._reset_fit()
        self.fit_lower = None
        self.fit_upper = None
    def _transform(self, val):
        if not self.fit_complete:
            raise ValueError("StandardScaler has not been fit yet.")
        if self.nullable and (val is None or isinstance(val, float) and np.isnan(val)):
            return (0.0, 1)
        else:
            val = (val - self.mean) / np.sqrt(self.var)
        if self.nullable:
            return (val, 0)
        else:
            return val
    def _inverse_transform(self, val):
        if not self.fit_complete:
            raise ValueError("StandardScaler has not been fit yet.")
        if self.nullable:
            v, n = val
            val = v
            if n == 1:
                return None
        val = val * np.sqrt(self.var) + self.mean
        return np.clip(val, self.fit_lower, self.fit_upper)
