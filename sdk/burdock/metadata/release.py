"""
Represents the metadata for a differentially private release.

We currently implement a subset of PSI release metadata.  This
will be updated to align with yarrow after yarrow enhanced release
metadata format is specified.
"""
class Result:
    def __init__(self, statistic, mechanism, name, epsilon, delta, release, intervals):
        """Instantiate a release.

        :param string statistic: The label for the statistic being computed (e.g. 'sum', 'mean')
        :param string mechanism: The label for the mechanism being used (e.g. 'laplace', 'gaussian')
        :param string name: The name for the variable, usually the column name
        :param float epsilon: The epsilon that was used in computing this value
        :param float delta: The delta privacy parameter used
        :param string[] release: Vector of noisy release values as string
        :param ((float, float, float)[]) intervals: Lower and upper bounds of confidence intervals at each pct.
         By default, only 95% is returned, but caller can request multiple.  The interval
         returned is centered on 0, and should not be thought of as an error bound.
        """
        self.statistic = statistic
        self.mechanism = mechanism
        self.name = name
        self.epsilon = epsilon
        self.delta = delta
        self.release = ([str(r) for r in release])
        self.intervals = intervals