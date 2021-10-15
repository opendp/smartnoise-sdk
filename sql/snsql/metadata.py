import yaml
import io
from os import path

from snsql.sql.reader.base import NameCompare

# implements spec at https://docs.google.com/document/d/1Q4lUKyEu2W9qQKq6A0dbo0dohgSUxitbdGhX97sUNOM/


class Metadata:
    """Information about a collection of tabular data sources"""

    def __init__(self, tables, engine, compare=None):
        """Instantiate a metadata object with information about tabular data sources

        :param tables: A list of Table descriptions
        :param engine: The name of the database engine used to query these tables.  Used for engine-
            specific name escaping and comparison.  Set to None to use default semantics.
        """
        self.m_tables = dict([(t.table_name(), t) for t in tables])
        self.engine = engine if engine is not None else "Unknown"
        self.compare = NameCompare.get_name_compare(engine) if compare is None else compare

    def __getitem__(self, tablename):
        schema_name = ""
        parts = tablename.split(".")
        if len(parts) == 2:
            schema_name, tablename = parts
        for tname in self.m_tables.keys():
            table = self.m_tables[tname]
            if self.compare.schema_match(
                schema_name, table.schema
            ) and self.compare.identifier_match(tablename, table.name):
                table.compare = self.compare
                return table
        return None

    def __str__(self):
        return "\n\n".join([str(self.m_tables[table]) for table in self.m_tables.keys()])

    def tables(self):
        return [self.m_tables[tname] for tname in self.m_tables.keys()]

    def __iter__(self):
        return self.tables()

    @staticmethod
    def from_file(file):
        """Load the metadata about this collection from a YAML file"""
        ys = CollectionYamlLoader(file)
        return ys.read_file()

    @staticmethod
    def from_dict(schema_dict):
        """Load the metadata from a dict object"""
        ys = CollectionYamlLoader("dummy")
        return ys._create_metadata_object(schema_dict)

    @classmethod 
    def from_(cls, val):
        if isinstance(val, Metadata):
            return val
        elif isinstance(val, str):
            if path.exists(val):
                return cls.from_file(val)
            else:
                # could allow YAML strings here.  Throw for now
                raise ValueError(f"Unable to load metadata path {val}")
        elif isinstance(val, dict):
            return cls.from_dict(val)
        else:
            raise ValueError(f"Metadata needs to be string, dictionary, or Metadata.  Got {str(type(val))}")

    def to_file(self, file, collection_name):
        """Save collection metadata to a YAML file"""
        ys = CollectionYamlLoader(file)
        ys.write_file(self, collection_name)

"""
    Common attributes for a table or a view
"""


class Table:
    """Information about a single tabular data source"""

    def __init__(
        self,
        schema,
        name,
        columns,
        rowcount=0,
        rows_exact=None,
        row_privacy=False,
        max_ids=1,
        sample_max_ids=True,
        clamp_counts=False,
        clamp_columns=True,
        use_dpsu=False,
        censor_dims=True,
    ):
        """Instantiate information about a tabular data source.

        :param schema: The schema is the SQL-92 schema used for disambiguating table names.  See
            documentation for more information about schema search path and resolution.
        :param name: The table name that will be used by SQL queries to reference data
            in this table.
        :param rowcount: The rough number of rows in this table.  Should not be the exact number, and does not need to be accurate
        :param columns: A list of Column objects with information about each column in the table.
        """
        self.schema = schema
        self.name = name
        self.rowcount = rowcount
        self.row_privacy = row_privacy
        self.max_ids = max_ids
        self.sample_max_ids = sample_max_ids
        self.rows_exact = rows_exact
        self.use_dpsu = use_dpsu
        self.clamp_counts = clamp_counts
        self.clamp_columns = clamp_columns
        self.censor_dims = censor_dims

        self.m_columns = dict([(c.name, c) for c in columns])
        self.compare = None

    def __getitem__(self, colname):
        for cname in self.m_columns.keys():
            col = self.m_columns[cname]
            if self.compare is None:
                # the database will attach the engine-specific comparer usually
                self.compare = NameCompare()
            if self.compare.identifier_match(colname, col.name):
                return col
        return None

    def __str__(self):
        return (
            str(self.schema)
            + "."
            + str(self.name)
            + " ["
            + str(self.rowcount)
            + " rows]\n\t"
            + "\n\t".join([str(self.m_columns[col]) for col in self.m_columns.keys()])
        )

    def key_cols(self):
        return [
            self.m_columns[name]
            for name in self.m_columns.keys()
            if self.m_columns[name].is_key == True
        ]

    def columns(self):
        return [self.m_columns[name] for name in self.m_columns.keys()]

    def __iter__(self):
        return self.columns()

    def table_name(self):
        return (self.schema + "." if len(self.schema.strip()) > 0 else "") + self.name


class String:
    """A column with string data"""

    def __init__(self, name, card, is_key=False, bounded=False):
        self.name = name
        self.card = card
        self.is_key = is_key
        self.bounded = bounded

    def __str__(self):
        return ("*" if self.is_key else "") + str(self.name) + " (card: " + str(self.card) + ")"

    def typename(self):
        return "string"


class Boolean:
    """A column with True/False data"""

    def __init__(self, name, is_key=False, bounded=False):
        self.name = name
        self.is_key = is_key
        self.bounded = bounded

    def __str__(self):
        return ("*" if self.is_key else "") + str(self.name) + " (boolean)"

    def typename(self):
        return "boolean"


class DateTime:
    """A date/time column"""

    def __init__(self, name, is_key=False, bounded=False):
        self.name = name
        self.is_key = is_key
        self.bounded = bounded

    def __str__(self):
        return ("*" if self.is_key else "") + str(self.name) + " (datetime)"

    def typename(self):
        return "datetime"


class Int:
    """A column with integer data"""

    def __init__(self, name, minval=None, maxval=None, is_key=False, bounded=False):
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
    """A floating point column"""

    def __init__(self, name, minval=None, maxval=None, is_key=False, bounded=False):
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
    """Column is unknown type.  Will be ignored.  May not be used in queries."""

    def __init__(self, name):
        self.name = name
        self.is_key = False

    def __str__(self):
        return str(self.name) + " (unknown)"

    def typename(self):
        return "unknown"


class CollectionYamlLoader:
    def __init__(self, file):
        self.file = file

    def read_file(self):
        if isinstance(self.file, io.IOBase):
            try:
                c_s = yaml.safe_load(self.file)
            except yaml.YAMLError as exc:
                raise
            return self._create_metadata_object(c_s)
        else:
            with open(self.file, "r") as stream:
                try:
                    c_s = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise
            return self._create_metadata_object(c_s)

    def _create_metadata_object(self, c_s):
        keys = list(c_s.keys())
        engine = "Unknown"

        if len(keys) == 0:
            raise ValueError("No collections in YAML file!")

        if len(keys) == 1:
            collection = keys[0]
        elif len(keys) > 2:
            raise ValueError("Please include only one collection per config file: " + str(keys))
        else:  # we have two keys; one should be engine
            if "engine" not in keys:
                raise ValueError("Please include only one collection per config file: " + str(keys))
            engine = c_s["engine"]
            collection = [k for k in keys if k != "engine"][0]

        db = c_s[collection]

        tables = []

        for schema in db.keys():
            s = db[schema]
            for table in s.keys():
                t = s[table]
                tables.append(self.load_table(schema, table, t))

        return Metadata(tables, engine)

    def load_table(self, schema, table, t):
        rowcount = int(t["rows"]) if "rows" in t else 0
        rows_exact = int(t["rows_exact"]) if "rows_exact" in t else None
        row_privacy = bool(t["row_privacy"]) if "row_privacy" in t else False
        max_ids = int(t["max_ids"]) if "max_ids" in t else 1
        sample_max_ids = bool(t["sample_max_ids"]) if "sample_max_ids" in t else True
        use_dpsu = bool(t["use_dpsu"]) if "use_dpsu" in t else False
        clamp_counts = bool(t["clamp_counts"]) if "clamp_counts" in t else False
        clamp_columns = bool(t["clamp_columns"]) if "clamp_columns" in t else True
        censor_dims = bool(t["censor_dims"]) if "censor_dims" in t else True

        columns = []
        colnames = [
            cn
            for cn in t.keys()
            if cn
            not in [
                "rows",
                "rows_exact",
                "row_privacy",
                "max_ids",
                "sample_max_ids",
                "clamp_counts",
                "clamp_columns",
                "use_dpsu",
                "censor_dims",
            ]
        ]
        for column in colnames:
            columns.append(self.load_column(column, t[column]))

        return Table(
            schema,
            table,
            columns,
            rowcount,
            rows_exact,
            row_privacy,
            max_ids,
            sample_max_ids,
            clamp_counts,
            clamp_columns,
            use_dpsu,
            censor_dims,
        )

    def load_column(self, column, c):
        is_key = False if "private_id" not in c else bool(c["private_id"])
        bounded = False if "bounded" not in c else bool(c["bounded"])

        if c["type"] == "boolean":
            return Boolean(column, is_key, bounded)
        elif c["type"] == "datetime":
            return DateTime(column, is_key, bounded)
        elif c["type"] == "int":
            minval = int(c["lower"]) if "lower" in c else None
            maxval = int(c["upper"]) if "upper" in c else None
            return Int(column, minval, maxval, is_key, bounded)
        elif c["type"] == "float":
            minval = float(c["lower"]) if "lower" in c else None
            maxval = float(c["upper"]) if "upper" in c else None
            return Float(column, minval, maxval, is_key, bounded)
        elif c["type"] == "string":
            card = int(c["cardinality"]) if "cardinality" in c else 0
            return String(column, card, is_key, bounded)
        else:
            raise ValueError("Unknown column type for column {0}: {1}".format(column, c))

    def write_file(self, collection_metadata, collection_name):

        engine = collection_metadata.engine
        schemas = {}
        for t in collection_metadata.tables():
            schema_name = t.schema
            table_name = t.name
            if schema_name not in schemas:
                schemas[schema_name] = {}
            schema = schemas[schema_name]
            if table_name in schema:
                raise ValueError(
                    "Attempt to insert table with same name twice: " + schema_name + table_name
                )
            schema[table_name] = {}
            table = schema[table_name]
            table["rows"] = t.rowcount
            if t.row_privacy is not None:
                table["row_privacy"] = t.row_privacy
            if t.max_ids is not None:
                table["max_ids"] = t.max_ids
            if t.sample_max_ids is not None:
                table["sample_max_ids"] = t.sample_max_ids
            if t.rows_exact is not None:
                table["rows_exact"] = t.rows_exact
            if t.use_dpsu is not None:
                table["use_dpsu"] = t.use_dpsu
            if t.clamp_counts is not None:
                table["clamp_counts"] = t.clamp_counts
            if t.clamp_columns is not None:
                table["clamp_columns"] = t.clamp_columns
            if t.censor_dims is not None:
                table["censor_dims"] = t.censor_dims

            for c in t.columns():
                cname = c.name
                if cname in table:
                    raise ValueError(
                        "Duplicate column name {0} in table {1}".format(cname, table_name)
                    )
                table[cname] = {}
                column = table[cname]
                if hasattr(c, "bounded") and c.bounded == True:
                    column["bounded"] = c.bounded
                if hasattr(c, "card"):
                    column["cardinality"] = c.card
                if hasattr(c, "minval") and c.minval is not None:
                    column["lower"] = c.minval
                if hasattr(c, "maxval") and c.maxval is not None:
                    column["upper"] = c.maxval
                if c.is_key is not None and c.is_key == True:
                    column["private_id"] = c.is_key
                if type(c) is String:
                    column["type"] = "string"
                elif type(c) is Int:
                    column["type"] = "int"
                elif type(c) is Float:
                    column["type"] = "float"
                elif type(c) is Boolean:
                    column["type"] = "boolean"
                elif type(c) is DateTime:
                    column["type"] = "datetime"
                elif type(c) is Unknown:
                    column["type"] = "unknown"
                else:
                    raise ValueError("Unknown column type: " + str(type(c)))
        db = {}
        db[collection_name] = schemas
        db["engine"] = collection_metadata.engine
        if isinstance(self.file, io.IOBase):
            raise ValueError("Cannot save metadata to a file stream.  Please use file path")
        with open(self.file, "w") as outfile:
            yaml.dump(db, outfile)
