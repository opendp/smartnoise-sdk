from snsql.metadata import Metadata
from snsql.sql.reader.postgres import PostgresNameCompare
import io

meta_str_db_escaped_schema = """
engine: postgres
My-DB:
  "`My-Schema`":
    My-Table:
      row_privacy: True
      age:
        type: int
"""

meta_str_db = """
My-DB:
  My-Schema:
    My-Table:
      row_privacy: True
      age:
        type: int
"""

meta_str = """
"":
  My-Schema:
    My-Table:
      row_privacy: True
      age:
        type: int
"""

meta_str_no_schema = """
"":
  "":
    My-Table:
      row_privacy: True
      age:
        type: int
"""


class TestTableNameMatch:
    def test_with_dbname_escaped_schema(self):
        meta = Metadata.from_(io.StringIO(meta_str_db_escaped_schema))
        # we don't set compare here, since engine is specified in metadata
        assert(meta["My-DB.`My-Schema`.My-Table"])
        assert(meta["my-db.`My-Schema`.My-Table"])
        assert(not meta["Database.`My-Schema`.My-Table"])
        assert(meta["`My-Schema`.My-Table"])
    def test_with_dbname(self):
        meta = Metadata.from_(io.StringIO(meta_str_db))
        meta.compare = PostgresNameCompare()
        assert(meta["My-DB.My-Schema.My-Table"])
        assert(meta["my-db.My-Schema.My-Table"])
        assert(not meta["my-db.`My-Schema`.My-Table"])
        assert(not meta["Database.`My-Schema`.My-Table"])
        assert(meta["My-Schema.My-Table"])
    def test_no_db(self):
        meta = Metadata.from_(io.StringIO(meta_str))
        meta.compare = PostgresNameCompare()
        assert(meta["Database.My-Schema.My-Table"])
        assert(meta["My-DB.My-Schema.My-Table"])
        assert(meta["my-db.My-Schema.My-Table"])
        assert(not meta["my-db.`My-Schema`.My-Table"])
        assert(meta["My-Schema.My-Table"])
    def test_no_schema(self):
        meta = Metadata.from_(io.StringIO(meta_str_no_schema))
        meta.compare = PostgresNameCompare()
        assert(not meta["My-DB.My-Schema.My-Table"])
        assert(meta["My-Table"])