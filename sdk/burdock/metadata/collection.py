import importlib
from .name_compare import BaseNameCompare

# implements spec at https://docs.google.com/document/d/1Q4lUKyEu2W9qQKq6A0dbo0dohgSUxitbdGhX97sUNOM/

class Collection:
    def __init__(self, tables, engine, compare=None):
        self.m_tables = dict([(t.table_name(), t) for t in tables])
        self.engine = engine if engine is not None else "Unknown"
        if compare is None:
            if engine == "Unknown":
                self.compare = BaseNameCompare()
            else:
                engine_module_name = "burdock.query.sql.reader." + engine.lower() + "_reader"
                compare_class = engine + "NameCompare"
                try:
                    engine_module = importlib.import_module(engine_module_name)
                    name_compare = getattr(engine_module, compare_class)
                    self.compare = name_compare()
                except Exception:
                    self.compare = BaseNameCompare()

    def __getitem__(self, tablename):
        schema_name = ''
        parts = tablename.split('.')
        if len(parts) == 2:
            schema_name, tablename = parts
        for tname in self.m_tables.keys():
            table = self.m_tables[tname]
            if self.compare.schema_match(schema_name, table.schema) and self.compare.identifier_match(tablename, table.name):
                table.compare = self.compare
                return table
        return None
    def __str__(self):
        return "\n\n".join([str(self.m_tables[table]) for table in self.m_tables.keys()])
    def tables(self):
        return [self.m_tables[tname] for tname in self.m_tables.keys()]


"""
    Common attributes for a table or a view
"""
class Table:
    def __init__(self, schema, name, rowcount, columns, row_privacy=False, max_ids=None, sample_max_ids=True, rows_exact=None):
        self.schema = schema
        self.name = name
        self.rowcount = rowcount
        self.row_privacy = row_privacy
        self.max_ids = max_ids
        self.sample_max_ids = sample_max_ids
        self.rows_exact = rows_exact
        self.m_columns = dict([(c.name, c) for c in columns])
        self.compare = None
    def __getitem__(self, colname):
        for cname in self.m_columns.keys():
            col = self.m_columns[cname]
            if self.compare is None:
                # the database will attach the engine-specific comparer usually
                self.compare = BaseNameCompare()
            if self.compare.identifier_match(colname, col.name):
                return col
        return None
    def __str__(self):
        return str(self.schema) + "." + str(self.name) + " [" + str(self.rowcount) + " rows]\n\t" + "\n\t".join([str(self.m_columns[col]) for col in self.m_columns.keys()])
    def key_cols(self):
        return [self.m_columns[name] for name in self.m_columns.keys() if self.m_columns[name].is_key == True]
    def columns(self):
        return [self.m_columns[name] for name in self.m_columns.keys()]
    def table_name(self):
        return (self.schema + "." if len(self.schema.strip()) > 0 else "") + self.name

class String:
    def __init__(self, name, card, is_key = False, bounded = False):
        self.name = name
        self.card = card
        self.is_key = is_key
        self.bounded = bounded
    def __str__(self):
        return ("*" if self.is_key else "") + str(self.name) + " (card: " + str(self.card) + ")"
    def typename(self):
        return "string"

class Boolean:
    def __init__(self, name, is_key = False, bounded = False):
        self.name = name
        self.is_key = is_key
        self.bounded = bounded
    def __str__(self):
        return ("*" if self.is_key else "") + str(self.name) + " (boolean)"
    def typename(self):
        return "boolean"

class DateTime:
    def __init__(self, name, is_key = False, bounded = False):
        self.name = name
        self.is_key = is_key
        self.bounded = bounded
    def __str__(self):
        return ("*" if self.is_key else "") + str(self.name) + " (datetime)"
    def typename(self):
        return "datetime"

class Int:
    def __init__(self, name, minval = None, maxval = None, is_key = False, bounded = False):
        self.name = name
        self.minval = minval
        self.maxval = maxval
        self.is_key = is_key
        self.unbounded = minval is None or maxval is None
        self.bounded = bounded
    def __str__(self):
        bounds = "unbounded" if self.unbounded else str(self.minval) + "," + str(self.maxval) 
        return ("*" if self.is_key else "") + str(self.name) + " [int] (" + bounds + ")"
    def typename(self):
        return "int"

class Float:
    def __init__(self, name, minval = None, maxval = None, is_key = False, bounded = False):
        self.name = name
        self.minval = minval
        self.maxval = maxval
        self.is_key = is_key
        self.unbounded = minval is None or maxval is None
        self.bounded = bounded
    def __str__(self):
        bounds = "unbounded" if self.unbounded else str(self.minval) + "," + str(self.maxval) 
        return ("*" if self.is_key else "") + str(self.name) + " [float] (" + bounds + ")"
    def typename(self):
        return "float"

class Unknown:
    def __init__(self, name):
        self.name = name
        self.is_key = False
    def __str__(self):
        return str(self.name) + " (unknown)"
    def typename(self):
        return "unknown"


