
class ColumnTransformer:
    cache_fit = True
    def __init__(self):
        self._fit_complete = False
        self._fit_vals = []
    def fit(self, data, idx=None):
        # helper method is useful when fitting a single column
        self.fit = False
        if idx is None:
            for val in data:
                self._fit(val)
        else:
            for row in data:
                self._fit(row[idx])
        self._fit_finish()
    def transform(self, data, idx=None):
        if idx is None:
            return [self._transform(val) for val in data]
        else:
            return [self._transform(row[idx]) for row in data]
    def fit_transform(self, data, idx=None):
        self.fit(data, idx)
        return self.transform(data, idx)
    def inverse_transform(self, data, idx=None):
        if idx is None:
            return [self._inverse_transform(val) for val in data]
        else:
            return [self._inverse_transform(row[idx]) for row in data]
    def _fit(self, val):
        # fits a single value
        if self.cache_fit:
            self._fit_vals.append(val)
    def _fit_finish(self):
        # called after all values have been fit
        self._fit_complete = True
    def _transform(self, val):
        # transforms a single value
        raise NotImplementedError
    def _inverse_transform(self, val):
        # inverse transforms a single value
        raise NotImplementedError