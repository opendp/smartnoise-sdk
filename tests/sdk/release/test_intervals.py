import pytest
from opendp.smartnoise.report import Interval, Intervals


class TestIntervals:
    def test_interval(self):
        ival1 = Interval(0.95, 3.0)
        ival2 = Interval(0.77, 2.6)
        for i in range(5):
            ival1.append(i)
            ival2.append(i)
        assert(len(ival1) == 5)
        low, high = ival1[3]
        assert(low == 0.0)
        assert(high == 6.0)
        for iv1, iv2 in zip(ival1, ival2):
            assert(iv1.contains(iv2))
        ival1.remove(tuple([-1.0, 3.3]))  # remove nothing
        assert(len(ival1) == 5)
        ival1.remove(tuple([0.0, 6.0]))  # remove fourth item
        assert(len(ival1) == 4)
    def test_intervals_collection(self):
        vals = range(20)
        ival1 = Interval(0.95, 3.0)
        ival2 = Interval(0.77, 2.77)
        ival1.extend(vals)
        ival2.extend(vals)
        ival_a = Interval(0.95, 3.0)
        ival_b = Interval(0.77, 2.7)
        ivals = Intervals([ival_a, ival_b])
        assert(ivals.interval_widths == [0.95, 0.77])
        assert(ivals.accuracy == [3.0, 2.7])
        for a, b in zip(ival1, ival_a):
            assert(a == b)
        for a, b in zip(ival2, ival_b):
            assert(a == b)
