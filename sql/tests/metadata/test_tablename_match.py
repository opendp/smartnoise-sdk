from snsql.metadata import Metadata
from snsql.sql.privacy import Privacy
from snsql.sql.private_reader import PrivateReader
from snsql.sql.reader.postgres import PostgresNameCompare, PostgresReader
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
    "My-Table":
      row_privacy: True
      age:
        type: int
"""

meta_str_public_schema = """
"":
  "public":
    "My-Table":
      row_privacy: True
      age:
        type: int
"""

meta_str_mixed_escape = """
engine: postgres
'"My-Database"':
  "`My-Schema`":
    "`My-Table`":
      row_privacy: True
      person-age:
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

class TestRewriterTableNameMatch:
    def test_with_dbname_escaped_schema(self):
      meta = Metadata.from_(io.StringIO(meta_str_mixed_escape))
      query = 'SELECT SUM(`person-age`) FROM `My-Database`.[My-Schema]."My-Table"'
      privacy = Privacy(epsilon=1.0, delta=0.1)
      reader = PostgresReader()
      priv = PrivateReader(reader, meta, privacy=privacy)
      subq, _ = priv._rewrite(query)
      cols = subq.xpath('//Column')
      assert(len(cols) == 2)
      for col in cols:
          assert(col.name == '`person-age`')
      tab = subq.xpath_first('//Table')
      assert(tab.name == '`My-Database`.[My-Schema]."My-Table"')

class TestAstAttachMetadata:
  def test_no_schema(self):
    # metadata specifies no schema, and query specifies no schema
    meta = Metadata.from_(io.StringIO(meta_str_no_schema))
    query = 'SELECT SUM(age) FROM "my-table"'
    privacy = Privacy(epsilon=1.0)
    reader = PostgresReader()
    priv = PrivateReader(reader, meta, privacy=privacy)
    subq, _ = priv._rewrite(query)
    cols = subq.xpath('//Column')
    assert(len(cols) > 0)
    tab = subq.xpath_first('//Table')
    assert(tab.name == '"my-table"')
  def test_no_schema_public(self):
    # metadata specifies no schema, but query specifies public schema
    meta = Metadata.from_(io.StringIO(meta_str_no_schema))
    query = 'SELECT SUM(age) FROM public."my-table"'
    privacy = Privacy(epsilon=1.0)
    reader = PostgresReader()
    priv = PrivateReader(reader, meta, privacy=privacy)
    subq, _ = priv._rewrite(query)
    cols = subq.xpath('//Column')
    assert(len(cols) > 0)
    tab = subq.xpath_first('//Table')
    assert(tab.name == 'public."my-table"')
  def test_public_schema_public(self):
    # metadata specifies public schema, but query specifies no schema
    meta = Metadata.from_(io.StringIO(meta_str_public_schema))
    query = 'SELECT SUM(age) FROM "my-table"'
    privacy = Privacy(epsilon=1.0)
    reader = PostgresReader()
    priv = PrivateReader(reader, meta, privacy=privacy)
    subq, _ = priv._rewrite(query)
    cols = subq.xpath('//Column')
    assert(len(cols) > 0)
    tab = subq.xpath_first('//Table')
    assert(tab.name == '"my-table"')