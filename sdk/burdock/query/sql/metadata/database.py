from .metadata import *

from burdock.query.sql import QueryParser, ast
from burdock.query.sql.metadata import sql_reservered

class MetadataLoader:
    def __init__(self, reader):
        self.reader = reader
        self.dbname = reader.db_name()
        self.engine = reader.engine

    def databaseSchema(self, tables=None):
        return Collection(self.getTables(tables), self.engine)

    def getTables(self, tableList=None):
        if tableList is None:
            tableList = self.listTables()

        tables = [self.loadTableSchema(schema, name) for (schema, name) in tableList]
        return [t for t in tables if t.rowcount > 0]

    def listTables(self, schema=None):
        if schema is None:
            filter = "AND table_schema <> 'pg_catalog' AND table_schema <> 'information_schema'"
        else:
            filter = "AND table_schema = '{0}'".format(schema)

        sql = "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE' "
        sql += filter + ' '
        sql += " AND table_catalog='{0}';".format(self.dbname)
        return self.reader.execute(sql)[1:]

    def listViews(self):
        sql = "SELECT table_schema, table_name \
            FROM information_schema.views WHERE table_schema <> 'pg_catalog' AND table_schema <> 'information_schema' \
            AND table_catalog='{0}';".format(self.dbname)
        return self.reader.execute(sql)

    def listColumns(self, schema, table):
        sql = "SELECT COLUMN_NAME, DATA_TYPE \
            FROM INFORMATION_SCHEMA.COLUMNS WHERE \
            TABLE_SCHEMA='{0}' AND TABLE_NAME='{1}'".format(schema, table)
        return [(row[0], row[1]) for row in self.reader.execute(sql)[1:]]


    def loadTableSchema(self, schema, table):
        tableName = escape(schema) + "." + escape(table)
        columnList = self.listColumns(schema, table)
        rowcount, typedColumns = self.getTypedColumns(tableName, columnList)
        return Table(escape(schema), escape(table), rowcount, typedColumns)

    """
        Reads in all columns in a table schema result set and converts
        to the lowest common denominator type supported by the
        differential privacy platform.
    """
    def getTypedColumns(self, tableName, columnList):
        typedColumnList = [(name, self.colType(name, datatype)) for (name, datatype) in columnList ]
        stringCols = [name for (name, datatype) in typedColumnList if datatype == "string"]
        numberCols = [name for (name, datatype) in typedColumnList if datatype in ["int", "float"]]

        rowCount = [ast.NamedExpression("cg_nrows_count", ast.AggFunction("COUNT", None, ast.AllColumns(None)))]
        #stringCard = [ast.NamedExpression(name, ast.AggFunction("COUNT", "DISTINCT", ast.Column(escape(name)))) for name in stringCols]
        numberMin = [ast.NamedExpression(name + "_cgminval", ast.AggFunction("MIN", None, ast.Column(escape(name)))) for name in numberCols]
        numberMax = [ast.NamedExpression(name + "_cgmaxval", ast.AggFunction("MAX", None, ast.Column(escape(name)))) for name in numberCols]

        metadataCols = rowCount +  numberMax + numberMin # + stringCard

        if len(metadataCols) == 0:
            return []
        try:
            selectClause = ast.Select(None, metadataCols)
            fromClause = ast.From([ast.Relation(ast.Table(tableName, None), None)])
            query = ast.Query(selectClause, fromClause, None, None, None, None)
            metadataVals = self.reader.execute(str(query))[1]
            rowCount = metadataVals[0]
            nameVals = dict([(ne.name, val) for (ne, val) in zip(metadataCols, metadataVals)])
        except Exception as e:
            print("Error querying table metadata for " + tableName + "\n" + str(e))
            return (0, [])

        return (rowCount, [self.getTypedColumn(name, datatype, nameVals) for (name, datatype) in typedColumnList])

    def getTypedColumn(self, name, datatype, nameVals):
        if datatype == "string":
            #card = nameVals[name]
            return String(escape(name), 0)
        elif datatype == "int":
            minval = nameVals[name + "_cgminval"]
            maxval = nameVals[name + "_cgmaxval"]
            return Int(escape(name), minval, maxval)
        elif datatype == "float":
            minval = nameVals[name + "_cgminval"]
            maxval = nameVals[name + "_cgmaxval"]
            return Float(escape(name), minval, maxval)
        elif datatype == "datetime":
            return DateTime(escape(name))
        elif datatype == "boolean":
            return Boolean(escape(name))
        else:
            return Unknown(escape(name))

    def colType(self, name, dbtype):
        ints =["int", "smallint", "bigint", "integer",]
        floats = ["float", "double", "decimal", "double precision", "real"]
        dates = ["datetime", "abstime", "date", "time", "timestamp"]
        booleans = ["boolean", "bit"]                                                                                                                             
        strings = ["char","nchar","nvarchar", "varchar", "character varying", "text"]

        if any([dbtype == t or dbtype.startswith(t+'(') for t in ints]):
            return "int"
        elif any([dbtype == t or dbtype.startswith(t+'(') for t in floats]):
            return "float"
        elif any([dbtype == t or dbtype.startswith(t+'(') for t in strings]):
            return "string"
        elif any([dbtype == t or dbtype.startswith(t+'(') for t in booleans]):
            return "boolean"
        elif any([dbtype == t or dbtype.startswith(t+'(') for t in dates]):
            return "datetime"
        else:
            return "unknown"

def escape(value):
    if value.lower not in sql_reserved and value == value.lower():
        return value
    else:
        return ".".join(['"' + v + '"'for v in value.split('.') ])
