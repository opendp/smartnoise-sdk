import pytest
from opendp.whitenoise.report import Result, Report, Interval, Intervals

class TestRelease:
    def test_release(self):
        ival = Interval(0.9, 3.0)
        ival2 = Interval(0.7, 2.0)
        vals = range(100)
        ival.extend(vals)
        ival2.extend(vals)
        r = Result("foo", "bar", "baz", vals, 0.1, None, 1.0, None, 1, [ival, ival2] )
        assert(r.name == "baz")
        assert(len(r) == 100)
        del r[12]
        assert(len(r) == 99)
        for iv in r.intervals:
            assert(len(iv) == 99)
        del r[7]
        assert(len(r.intervals[0.9]) == 98)
        for a, b in zip(r.intervals[0.9], r.intervals[0.7]):
            assert(a.contains(b))

        vals_b = [r + 10000 for r in range(100)]
        ival_b = Interval(0.9, 3.0)
        r_b = Result("foo", "bar", "baz", vals_b, 0.1, None, 1.0, None, 1, [ival_b], "biff" )

        rel = Report([r, r_b])

        for a, b in zip(rel["baz"].intervals[0.9], rel["biff"].intervals[0.9]):
            assert a < b
