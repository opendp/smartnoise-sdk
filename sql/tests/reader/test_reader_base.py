import pytest

from snsql.reader.base import Reader
from snsql.sql.reader.engine import Engine
from snsql.sql.reader import BigQueryReader, PandasReader, PostgresReader, PrestoReader, SqlServerReader, SparkReader

READERS = BigQueryReader, PandasReader, PostgresReader, PrestoReader, SqlServerReader, SparkReader

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
    reader = Reader()
    assert reader.engine is None

    engine_override = "foo"

    class ReaderInherit(Reader):
        ENGINE = engine_override

    reader_inherit = ReaderInherit()
    assert reader_inherit.engine == engine_override
