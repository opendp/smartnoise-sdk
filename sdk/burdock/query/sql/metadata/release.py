"""
Represents the metadata for a differentially private release.

We currently implement a subset of PSI release metadata.  This
will be updated to align with yarrow after yarrow enhanced release
metadata format is specified.
"""
class Result:
    """
    Instantiate a release.

    Parameters:
        statistic (string): The label for the statistic being computed (e.g. 'sum', 'mean')
        mechanism (string): The label for the mechanism being used (e.g. 'laplace', 'gaussian')
        name (string): The name for the variable, usually the column name
        epsilon (float): The epsilon that was used in computing this value
        delta (float): The delta privacy parameter used
        release (string[]): Vector of noisy release values as string
        intervals ((pct, float, float)[]): Lower and upper bounds of confidence intervals at each pct.
            By default, only 95% is returned, but caller can request multiple.  The interval
            returned is centered on 0, and should not be thought of as an error bound.
    """
    def __init__(self, statistic, mechanism, name, epsilon, delta, release, intervals):
        self.statistic = statistic
        self.mechanism = mechanism
        self.name = name
        self.epsilon = epsilon
        self.delta = delta
        self.release = ([str(r) for r in release])
        self.intervals = intervals