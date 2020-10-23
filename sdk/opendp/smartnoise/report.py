import collections.abc
import numpy as np
"""
Represents the metadata for a differentially private release.

Specification at https://docs.google.com/document/d/1PTAG2xImB5B3m4crc9t3MQLyRUp3GieD-u8tJlNCDzY/edit#heading=h.ga5nyepy7ehj

Note that SQL results may return multiple rows, with columns representing a vector of
values that share a common mechanism, statistic, source, epsilon, delta, interval_widths, and accuracy,
while having multiple values per column, and multiple intervals per value.

For single row queries with default single confidence interval, there will be only one value
per column and one interval per value, but callers should be prepared for multiple values.
"""

class IntervalRange:
    """A low and high for a single value interval.  Used for row-based access."""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __str__(self):
        return "[{0}-{1}]".format(round(self.low, 2), round(self.high, 2))

    def contains(self, other):
        return self.low <= other.low and self.high >= other.high

    def inside(self, other):
        return other.contains(self)

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def __lt__(self, other):
        return self.high < other.low

    def __gt__(self, other):
        return self.low > other.high

    def intersects(self, other):
        return (self.low >= other.low and self.low <= other.high) or (self.high >= other.low and self.high <= other.high)

    def __iter__(self):
        return iter([self.low, self.high])


class Interval:
    """A vector of CIs for a single column and confidence."""

    def __init__(self, confidence, accuracy, low=None, high=None):
        """Collection of confidence intervals for a given width

        :param float confidence: the confidence interval width for the CIs in the list.  Between 0.0 and 1.0 inclusive.
        :param float accuracy: the plus/minus accuracy for values generated here. May be None, if not known or not symmetrical
        :param float[] low: the lower bound for each CI in the list
        :param float[] high: the lower bound for each CI in the list
        """
        self.low = low
        self.high = high
        if high is None:
            self.low = []
            self.high = []
        self.confidence = confidence
        self.accuracy = accuracy

    def __str__(self):
        cis = ", ".join(["[{0}-{1}]".format(round(low, 2), round(high, 2)) for low, high in zip(self.low, self.high)])
        return "confidence: {0}\naccuracy: {1}\n".format(self.confidence, self.accuracy) + cis

    def __len__(self):
        return len(self.low)

    def contains(self, other):
        return all([s.contains(o) for s, o in zip(self, other)])

    def inside(self, other):
        return all([s.inside(o) for s, o in zip(self, other)])

    def __eq__(self, other):
        return all([s == o for s, o in zip(self, other)])

    def __lt__(self, other):
        return all([s < o for s, o in zip(self, other)])

    def __gt__(self, other):
        return all([s > o for s, o in zip(self, other)])

    def intersects(self, other):
        return all([s.intersects(o) for s, o in zip(self, other)])
    def __iter__(self):
        return iter([IntervalRange(low, high) for low, high in zip(self.low, self.high)])
    def __getitem__(self, key):
        if isinstance(key, int):
            return IntervalRange(self.low[key], self.high[key])
        else:
            raise ValueError("Invalid index to interval: " + key)
    def __setitem__(self, key, value):
        if isinstance(key, int):
            if isinstance(value, tuple) or isinstance(value, IntervalRange):
                low, high = value
            else:
                if isinstance(value, float) or isinstance(value, int):
                    if self.accuracy is not None:
                        low = value * 1.0 - self.accuracy
                        high = value * 1.0 + self.accuracy
                    else:
                        raise ValueError("Can only auto-convert value to interval if accuracy is set")
                else:
                    raise ValueError("Can only set interval value with a tuple or numeric value")
            self.low[key] = low
            self.high[key] = high
        else:
            raise ValueError("Invalid index to interval: " + key)

    def __delitem__(self, key):
        if isinstance(key, int):
            del self.low[key]
            del self.high[key]
        else:
            # note we don't support slicers
            raise ValueError("Invalid index to interval: " + key)

    def append(self, value):
        if isinstance(value, tuple) or isinstance(value, IntervalRange):
            low, high = value
            self.low.append(low)
            self.high.append(high)
        else:
            if isinstance(value, float) or isinstance(value, int):
                if self.accuracy is not None:
                    low = value * 1.0 - self.accuracy
                    high = value * 1.0 + self.accuracy
                    self.low.append(low)
                    self.high.append(high)
                else:
                    raise ValueError("Cannot automatically convert values to intervals if accuracy not symmetrical")
            else:
                raise ValueError("Can only append intervals as tuples, or automatically convert values.")        

    def extend(self, values):
        if isinstance(values, (collections.abc.Sequence, np.ndarray)) and not type(values) is str:
            for v in values:
                self.append(v)
        else:
            print(type(values))
            raise ValueError("Interval only supports extending intervals by auto-converting values")

    def pop(self, item=None):
        if item is None:
            low = self.low.pop()
            high = self.high.pop()
        else:
            low = self.low.pop(item)
            high = self.high.pop(item)
        return IntervalRange(low, high)

    def clear(self):
        self.low.clear()
        self.high.clear()

    def remove(self, value):
        low, high = value
        for idx in range(len(self.low)):
            if self.low[idx] == low and self.high[idx] == high:
                del self.low[idx]
                del self.high[idx]
                break


class Intervals:
    """Collection of confidence intervals for varying interval_widths.

    Column-vector based access to CIs, with helper methods for
    row-based manipulation."""

    def __init__(self, intervals):
        self._intervals = {}
        for i in intervals:
            self._intervals[i.confidence] = i

    def __str__(self):
        return "Intervals: \n" + "\n".join(str(self._intervals[confidence]) for confidence in self._intervals.keys() )

    def __iter__(self):
        return iter([self._intervals[k] for k in self._intervals.keys()])

    def __getitem__(self, key):
        return self._intervals[key]

    def __setitem__(self, key, value):
        self._intervals[key] = value

    def __delitem__(self, key):
        del self._intervals[key]

    def keys(self):
        return self._intervals.keys()

    def append(self, value):
        if isinstance(value, int):
            for k in self._intervals.keys():
                self._intervals[k].append(value)
        else:
            raise ValueError("Interval collection only supports appending intervals by auto-converting values")

    def extend(self, values):
        if isinstance(values, (collections.abc.Sequence, np.ndarray)) and type(values) is not str:
            for k in self._intervals.keys():
                self._intervals[k].extend(values)
        else:
            raise ValueError("Interval collection only supports extending intervals by auto-converting values")

    def clear(self):
        for k in self._intervals.keys():
            self._intervals[k].clear()

    def delete_row(self, idx):
        for k in self._intervals.keys():
            interval = self._intervals[k]
            del interval[idx]

    @property
    def interval_widths(self):
        return [self._intervals[k].confidence for k in self._intervals.keys()]

    @property
    def alphas(self):
        return [1 - confidence for confidence in self.interval_widths]

    @property
    def accuracy(self):
        return [self._intervals[k].accuracy for k in self._intervals.keys()]

class Result:
    """A differentially private result, representing a single value or column of related values.

    Allows access to values and confidence intervals in column-vector format.  Helper
    methods for adding and deleting rows that span values and confidence intervals."""

    def __init__(self, mechanism, statistic, source, values, epsilon, delta, sensitivity, scale, max_contrib, intervals, name=None):
        """A result within a report.

        :param string mechanism: The label for the mechanism being used (e.g. 'laplace', 'gaussian')
        :param string statistic: The label for the statistic being computed (e.g. 'sum', 'mean')
        :param string source: The name or source expression for the variable, usually the column name
        :param object[] values:  An array of differentialy private values
        :param float epsilon: The epsilon that was used in computing this value
        :param float delta: The delta privacy parameter used
        :param float sensitivity: The sensitivity of the values
        :param float scale: The mechanism-specific noise level added by the mechanim. For example, may be variance for Gaussian, or scale for Laplace
        :param int max_contrib: The max contribution of individuals
        :param Intervals intervals: Typed collection of confidence intervals
        :param string name: A friendly name for the result.  If missing, uses the source.
        """
        self.mechanism = mechanism
        self.statistic = statistic
        self.source = source
        self.values = [v for v in values]
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.scale = scale
        self.max_contrib = max_contrib
        if isinstance(intervals, Intervals):
            self.intervals = intervals
        elif isinstance(intervals, list):
            self.intervals = Intervals(intervals)
        elif intervals is None:
            self.intervals = None
        else:
            raise ValueError("Don't know how to set intervals: " + str(intervals))
        self.name = name if name is not None else source

    def __str__(self):
        return "mechanism: {0}\nstatistic: {1}\nsource: {2}\nvalues: {3}\nepsilon: {4}\ndelta: {5}\nsensitivity: {6}\nmax_contrib: {7}\nintervals: {8}".format(self.mechanism, self.statistic, self.source, self.values, self.epsilon, self.delta, self.sensitivity, self.max_contrib, self._intervals)

    def __delitem__(self, idx):
        del self.values[idx]
        if self.intervals is not None:
            self.intervals.delete_row(idx)

    def __len__(self):
        return len(self.values)

    @property
    def interval_widths(self):
        return None if self.intervals is None else self.intervals.interval_widths

    @property
    def alphas(self):
        return None if self.intervals is None else self.intervals.alphas

    @property
    def accuracy(self):
        return None if self.intervals is None else self.intervals.accuracy


class Report:
    """A differentially private report

    Represents a list of result objects, with each result being
    a vector of differentially private results sharing common
    source and privacy paramaters.

    The individual result objects need not be the same length,
    though lengths will be the same for multi-column SQL outputs."""

    def __init__(self, results=None):
        self._results = {}
        if results is not None:
            for r in results:
                self._results[r.name] = r

    def __getitem__(self, key):
        return self._results[key]

    def __setitem__(self, key, value):
        self._results[key] = value

    def __delitem__(self, key):
        del self._results[key]

    def __contains__(self, key):
        return key in self._results
