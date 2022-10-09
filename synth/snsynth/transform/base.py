
from snsynth.transform.definitions import ColumnType


class ColumnTransformer:
    """Base class for column transformers.  Subclasses must implement the
    _fit, _transform, and _inverse_transform methods."""
    def __init__(self):
        self._fit_complete = False
        self.output_width = 0
        self._clear_fit()
    @property
    def output_type(self):
        """Must be implemented by subclasses to return the type of the output
        of the transformer.  From definitions.ColumnType enum."""
        raise NotImplementedError
    @property
    def is_categorical(self):
        return self.output_type == ColumnType.CATEGORICAL
    @property
    def is_continuous(self):
        return self.output_type == ColumnType.CONTINUOUS
    @property
    def fit_complete(self):
        return self._fit_complete
    @property
    def needs_epsilon(self):
        """Overriden by subclasses to indicate whether the transformer
        needs an epsilon value to be supplied to the fit method."""
        return False
    @property
    def cardinality(self):
        """Must be implemented by subclasses to return the cardinality of the
        output of the transformer.  Only applicable for categorical transformers."""
        return []
    def allocate_privacy_budget(self, epsilon, odometer):
        """Allocate privacy budget to the transformer.  This method is called
        by the DataTransformer to allocate privacy budget to the transformer.
        The default implementation does nothing, but subclasses can override
        this method to allocate privacy budget to the transformer.
        """
        pass
    def fit(self, data, idx=None):
        """Fit a column of data.  If data includes multiple columns,
        provide an index to select which column to fit.  If no index is supplied,
        data must be a single column as a list of values.

        .. code-block:: python

            t = MyColumnTransformer()
            assert(t.fit_complete == False)
            t.fit(iris, 0)  # fit the first column of iris
            assert(t.fit_complete == True)

        This method will always consume all rows in the data, even if the fit method
        has already been called.  This is to ensure consistency when passing iterators.
        If you want to ensure that fit only gets called when the transformer has not already
        been fit, you can check the fit_complete property.

        .. code-block:: python

            t = MyColumnTransformer()
            if not t.fit_complete:
                t.fit(iris, 0)
            assert(t.fit_complete == True)

        Calling this method repeatedly should perform a fresh fit each time using the new data.  
        """
        self._clear_fit()
        if idx is None:
            for val in data:
                self._fit(val)
        else:
            for row in data:
                self._fit(row[idx])
        self._fit_finish()
    def transform(self, data, idx=None):
        """Transform a column of data.  If data includes multiple columns,
        provide an index to select which column to transform.  If no index is supplied,
        data must be a single column as a list of values.
        
        .. code-block:: python

            t = MyColumnTransformer()
            t.fit(iris, 0)  # fit the first column of iris
            iris_encoded = t.transform(iris, 0)  # transform the first column of iris
        """
        if idx is None:
            return [self._transform(val) for val in data]
        else:
            return [self._transform(row[idx]) for row in data]
    def fit_transform(self, data, idx=None):
        """Fit and transform a column of data.  If data includes multiple columns,
        provide an index to select which column to fit.  If no index is supplied,
        data must be a single column as a list of values.

        .. code-block:: python

            t = MyColumnTransformer()
            iris_encoded = t.fit_transform(iris, 0)  # fit and transform the first column of iris

        This method will always perform a fresh fit each time it is called.  A fit and a transform
        will each consume all rows in the data, so this method should be called only on a data structure
        that allows multiple passes over the data.
        """
        if not self.fit_complete:
            self.fit(data, idx)
        return self.transform(data, idx)
    def inverse_transform(self, data, idx=None):
        """Inverse-transform a column of data.  If data includes multiple columns,
        provide an index to select which column to inverse-transform.  If no index is supplied,
        data must be a single column as a list of values.
        
        .. code-block:: python

            t = MyColumnTransformer()
            t.fit(iris, 0)  # fit the first column of iris
            iris_encoded = t.transform(iris, 0)  # transform the first column of iris
            iris_decoded = t.inverse_transform(iris_encoded, 0)  # inverse-transform the first column of iris
            assert(all([a == b for a, b in zip(iris[0], iris_decoded)]))
            
        """
        if idx is None:
            return [self._inverse_transform(val) for val in data]
        else:
            return [self._inverse_transform(row[idx]) for row in data]
    def _fit(self, val):
        """Must be implemented by subclasses to fit a single value.
        """
        raise NotImplementedError
    def _fit_finish(self):
        """Should be implemented by subclasses if there is
        any work that needs to be done to finish the fit process."""
        self._fit_complete = True
    def _clear_fit(self):
        """Must be implemented by subclasses to clear any state that is
        dependent on the data that was fit.
        """
        raise NotImplementedError
    def _reset_fit(self):
        """Helper can be called by _clear_fit"""
        self._fit_complete = False
        self.output_width = 0
    def _transform(self, val):
        """Must be implemented by subclasses to transform a single value."""
        raise NotImplementedError
    def _inverse_transform(self, val):
        """Must be implemented by subclasses to inverse-transform a single value."""
        raise NotImplementedError

class CachingColumnTransformer(ColumnTransformer):
    """Base class for column transformers that cache their fit state.
    Subclasses must implement the _fit_finish, _transform and _inverse_transform methods.
    """
    def __init__(self):
        super().__init__()
        self._fit_vals = []
    def _fit(self, val):
        """Caches each fit value to be processed by _fit_finish.
        If this method is overridden, it should call super()._fit(val)"""
        self._fit_vals.append(val)
    def _reset_fit(self):
        """Helper can be called by _clear_fit"""
        super()._reset_fit()
        self._fit_vals = []
