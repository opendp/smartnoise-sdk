import pytest
import string
import numpy as np
import random
from burdock.query.sql.reader.rowset import TypedRowset

"""
Basic unit tests for typed rowset
"""

col1 = range(1000)
col2 = np.random.normal(7.23, 0.889, len(col1))
col3 = [''.join([random.choice(string.ascii_uppercase) for x in range(6)]) for y in col1]
col4 = [c % 6 for c in col1]

rows_1k = [('id', 'temp', 'code', 'bucket')] + [r for r in zip(col1, col2, col3, col4)]
rows_10 = rows_1k[0:11]
rows_1 = rows_1k[0:2]
types = ['int', 'float', 'string', 'int']
sens = [None, None, None, None]

class TestTypedRowset:
    def test_make_1k(self):
        trs = TypedRowset(rows_1k, types, sens)
        assert(len(trs) == 1000)
    def test_make_10(self):
        trs = TypedRowset(rows_10, types, sens)
        assert(len(trs) == 10)
    def test_make_1(self):
        trs = TypedRowset(rows_1, types, sens)
        assert(len(trs) == 1)
    def test_make_empty(self):
        trs = TypedRowset(rows_1[0:1], types, sens)
        assert(len(trs) == 0)
