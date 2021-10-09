import io
import pytest
from snsql.metadata import Metadata

class TestMetaBounds:
    def test_meta_from_string(self):
        meta_good = """col:
    ? ''
    : table:
        duration:
            type: float
        id:
            private_id: true
            type: int
        max_ids: 1
        row_privacy: false
        rows: 1000
        sample_max_ids: true
engine: pandas"""
        file = io.StringIO(meta_good)
        c = Metadata.from_file(file)
    def test_meta_bad_float(self):
        meta_bad_float = """col:
    ? ''
    : table:
        duration:
            type: float
        id:
            private_id: true
            type: int
        max_ids: 1
        row_privacy: false
        rows: 1000
        sample_max_ids: true
engine: pandas"""
        file = io.StringIO(meta_bad_float)
        c = Metadata.from_file(file)
        assert(c["table"]["duration"].unbounded)
    def test_meta_bad_int(self):
        meta_bad_float = """col:
    ? ''
    : table:
        id:
            private_id: true
            type: int
        events:
            type: int
        max_ids: 1
        row_privacy: false
        rows: 1000
        sample_max_ids: true
engine: pandas"""
        file = io.StringIO(meta_bad_float)
        c = Metadata.from_file(file)
        assert(c["table"]["events"].unbounded)


