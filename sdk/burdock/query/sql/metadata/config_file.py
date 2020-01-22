import yaml
from .metadata import *
class MetadataLoader:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def from_dict(schema_dict):
        return MetadataLoader("dummy")._create_schema(schema_dict)

    def read_schema(self):
        with open(self.filename, 'r') as stream:
            try:
                dbs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return self._create_schema(dbs)

    def _create_schema(self, dbs):
        keys = list(dbs.keys())
        engine = "Unknown"

        if len(keys) == 0:
            raise ValueError("No databases in YAML file!")

        if len(keys) == 1:
            database = keys[0]
        elif len(keys) > 2:
            raise ValueError("Please include only one database per config file: " + str(keys))
        else:  # we have two keys; one should be engine
            if 'engine' not in keys:
                raise ValueError("Please include only one database per config file: " + str(keys))
            engine = dbs['engine']
            database = [k for k in keys if k != 'engine'][0]

        db = dbs[database]

        tables = []

        for schema in db.keys():
            s = db[schema]
            for table in s.keys():
                t = s[table]
                tables.append(self.load_table(schema, table, t))

        return Database(tables, engine)

    def load_table(self, schema, table, t):
        rowcount = int(t["rows"]) if "rows" in t else 0

        columns = []
        colnames = [cn for cn in t.keys() if cn != "rows"]
        for column in colnames:
            columns.append(self.load_column(column, t[column]))

        return Table(schema, table, rowcount, columns)

    def load_column(self, column, c):
        if c["type"] == "boolean":
            return Boolean(column, False if "is_key" not in c else bool(c["is_key"]))
        elif c["type"] == "datetime":
            return DateTime(column, False if "is_key" not in c else bool(c["is_key"]))
        elif c["type"] == "int":
            minval = int(c["min"]) if "min" in c else None
            maxval = int(c["max"]) if "max" in c else None
            return Int(column, minval, maxval, False if "is_key" not in c else bool(c["is_key"]))
        elif c["type"] == "float":
            minval = float(c["min"]) if "min" in c else None
            maxval = float(c["max"]) if "max" in c else None
            return Float(column, minval, maxval, False if "is_key" not in c else bool(c["is_key"]))
        elif c["type"] == "string":
            card = int(c["cardinality"]) if "cardinality" in c else 0
            return String(column, card, False if "is_key" not in c else bool(c["is_key"]))
        else:
            raise ValueError("Unknown column type for column {0}: {1}".format(column, c))

    def write_schema(self, database, dbname):

        engine = database.engine
        schemas = {}
        for t in database.tables():
            schema_name = t.schema
            table_name = t.name
            if schema_name not in schemas:
                schemas[schema_name] = {}
            schema = schemas[schema_name]
            if table_name in schema:
                raise ValueError("Attempt to insert table with same name twice: " + schema_name + table_name)
            schema[table_name] = {}
            table = schema[table_name]
            table["rows"] = t.rowcount

            for c in t.columns():
                cname = c.name
                if cname in table:
                    raise ValueError("Duplicate column name {0} in table {1}".format(cname, table_name))
                table[cname] = {}
                column = table[cname]
                if c.is_key is not None and c.is_key == True:
                    column["is_key"] = c.is_key
                if type(c) is String:
                    column["type"] = "string"
                    column["cardinality"] = c.card
                elif type(c) is Int:
                    column["type"] = "int"
                    if hasattr(c, "minval") and c.minval is not None:
                        column["min"] = c.minval
                    if hasattr(c, "maxval") and c.maxval is not None:
                        column["max"] = c.maxval
                elif type(c) is Float:
                    column["type"] = "float"
                    if hasattr(c, "minval") and c.minval is not None:
                        column["min"] = c.minval
                    if hasattr(c, "maxval") and c.maxval is not None:
                        column["max"] = c.maxval
                elif type(c) is Boolean:
                    column["type"] = "boolean"
                elif type(c) is DateTime:
                    column["type"] = "datetime"
                elif type(c) is Unknown:
                    column["type"] = "unknown"
                else:
                    raise ValueError("Unknown column type: " + str(type(c)))
        db = {}
        db[dbname] = schemas
        db["engine"] = database.engine
        with open(self.filename, 'w') as outfile:
            yaml.dump(db, outfile)
