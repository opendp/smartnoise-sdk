import pytest

from burdock.reader.base import Reader
from burdock.sql.reader.engine import Engine
from burdock.sql.reader import PandasReader, PrestoReader, PostgresReader, SqlServerReader, SparkReader

READERS = PandasReader, PrestoReader, PostgresReader, SqlServerReader, SparkReader

"""
Test Coverage:
    - sql reader types override engine property
    - engine property is exposed as Reader().engine, Reader.ENGINE, and Reader.engine
"""


@pytest.mark.parametrize("cls", READERS)
def test_has_engine(cls):
    assert cls.ENGINE is not None
    assert cls.ENGINE in Engine.known_engines


def test_reader_has_engine_property():
    reader = Reader(None)
    assert reader.engine is None

    engine_override = "foo"

    class ReaderInherit(Reader):
        ENGINE = engine_override

    reader_inherit = ReaderInherit(None)
    assert reader_inherit.engine == engine_override
