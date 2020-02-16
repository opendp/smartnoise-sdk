"""
Represents the metadata for a differentially private release.

Specification at https://docs.google.com/document/d/1PTAG2xImB5B3m4crc9t3MQLyRUp3GieD-u8tJlNCDzY/edit#heading=h.ga5nyepy7ehj

Note that SQL results may return multiple rows, with columns representing a vector of
values that share a commoon mechanism, statistic, source, epsilon, delta, alphas, and accuracy,
while having multiple values per column, and multiple intervals per value.

For single row queries with default single confidence interval, there will be only one value
per column and one interval per value, but callers should be prepared for multiple values.
"""
class Result:
    def __init__(self, mechanism, statistic, source, value, epsilon, delta, sensitivity, max_contrib, alpha, accuracy, interval):
        """Instantiate a release.

        :param string mechanism: The label for the mechanism being used (e.g. 'laplace', 'gaussian')
        :param string statistic: The label for the statistic being computed (e.g. 'sum', 'mean')
        :param string source: The name or source expression for the variable, usually the column name
        :param object[] value:  An array of differentialy private values
        :param float epsilon: The epsilon that was used in computing this value
        :param float delta: The delta privacy parameter used
        :param float sensitivity: The sensitivity of the values
        :param int max_contrib: The max contribution of individuals
        :param float[] alpha:  A list of statistical significance levels for accuracy bounds.  By default, 0.05 is included, representing 95% confidence interval.
        :param float[][] accuracy:  A list of accuracy bounds for each alpha listed in alphas.
        :param ((float, float)[][]) interval: Lower and upper bounds of confidence intervals at each pct.
         By default, only 95% is returned, but caller can request multiple.
        """
        self.mechanism = mechanism
        self.statistic = statistic
        self.source = source
        self.value = value
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.max_contrib = max_contrib
        self.alpha = alpha
        self.accuracy = accuracy
        self.interval = interval