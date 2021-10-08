import pandas as pd
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model.base import LinearModel
from .dp_covariance import DPcovariance


class DPLinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Differentially Private Linear Regression using the Covariance Method.

    Fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, optional, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : float or array of shape of (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.



    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0000...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept

    def _set_coef_and_intercept(self, df):
        df_copy = df
        if self.fit_intercept:
            self.intercept_ = df_copy["intercept", "Estimate"]
            df_copy = df_copy.drop(["intercept"])
        self.coef_ = df_copy["Estimate"].values

    def fit(self, X, y, bounds, budget):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary


        Returns
        -------
        self : returns an instance of self.
        """
        data = pd.concat((X, y), axis=1)
        n = data.shape[0]

        cols = list(X.columns.values) + list(y.columns.values)

        results = DPcovariance(n, cols, bounds, budget).get_linear_regression(
            data, X.columns.values, y.columns.values, self.fit_intercept
        )
        self._set_coef_and_intercept(results)
        return self
