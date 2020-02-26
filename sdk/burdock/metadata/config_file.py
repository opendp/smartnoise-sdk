import yaml
from .collection import *


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

        return Collection(tables, engine)

    def load_table(self, schema, table, t):
        rowcount = int(t["rows"]) if "rows" in t else 0
        row_privacy = bool(t["row_privacy"]) if "row_privacy" in t else False
        max_ids = int(t["max_ids"]) if "max_ids" in t else 1
        sample_max_ids = bool(t["sample_max_ids"]) if "sample_max_ids" in t else None
        rows_exact = int(t["rows_exact"]) if "rows_exact" in t else None
        clamp_counts = bool(t["clamp_counts"]) if "clamp_counts" in t else True

        columns = []
        colnames = [cn for cn in t.keys() if cn != "rows"]
        for column in colnames:
            columns.append(self.load_column(column, t[column]))

        return Table(schema, table, rowcount, columns, row_privacy, max_ids, sample_max_ids, rows_exact, clamp_counts)

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
            if t.row_privacy is not None: 
                table["row_privacy"] = t.row_privacy
            if t.max_ids is not None: 
                table["max_ids"] = t.max_ids
            if t.sample_max_ids is not None: 
                table["sample_max_ids"] = t.sample_max_ids
            if t.rows_exact is not None:
                table["rows_exact"] = t.rows_exact
            if not t.clamp_counts:
                table["clamp_counts"] = t.clamp_counts

            for c in t.columns():
                cname = c.name
                if cname in table:
                    raise ValueError("Duplicate column name {0} in table {1}".format(cname, table_name))
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
        db[dbname] = schemas
        db["engine"] = database.engine
        with open(self.filename, 'w') as outfile:
            yaml.dump(db, outfile)
