from snsql.metadata import Metadata
from snsql.sql.parse import QueryParser

table_features = {
    "Store": {"name": "Store", "type": "int", "upper": 100, "lower": -50, "sensitivity": 150, "private_id": True},
    "Date": {"name": "Date", "type": "datetime"},
    "Temperature": {"name": "Temperature", "upper": 100, "lower": -50, "sensitivity": 75,  "type": "float"},
    "Fuel_Price": {"name": "Fuel_Price", "type": "float"},
    "IsHoliday": {"name": "IsHoliday", "type": "boolean", "missing_value": False},
    "row_privacy": False,
}

table_sales = {
    "Store": {"name": "Store", "type": "int", "private_id": True},
    "Date": {"name": "Date", "type": "datetime", "nullable": False},
    "Weekly_Sales": {"name": "Weekly_Sales", "type": "float"},
    "IsHoliday": {"name": "IsHoliday", "type": "boolean"},
    "row_privacy": False,
}

metadata_dict = {
    "engine": "sqlserver",
    "": {
        "": {
            "features": table_features,
            "sales": table_sales,
            }
        },
    }

metadata = Metadata.from_dict(metadata_dict)

class TestAstFromMeta:
    def test_meta_load(self):
        assert(metadata["sales"]["Date"].nullable == False)
        assert(metadata["features"]["Date"].nullable == True)
        assert(metadata["features"]["IsHoliday"].nullable == False)
        assert(metadata["features"]["Store"].sensitivity == 150)
        assert(metadata["features"]["Temperature"].sensitivity == 75)
    def test_ast_attach_sens(self):
        query = 'SELECT SUM("Temperature"), SUM(features."Store") AS store FROM features'
        q = QueryParser(metadata).query(query)
        assert(q._select_symbols[0].expression.sensitivity() == 75)
        assert(q._named_symbols['store'].expression.sensitivity() == 150)

        query = 'SELECT COUNT(DISTINCT "Temperature"), COUNT(features."Store") AS store FROM features'
        q = QueryParser(metadata).query(query)
        assert(q._select_symbols[0].expression.sensitivity() == 1)
        assert(q._named_symbols['store'].expression.sensitivity() == 1)
    def test_ast_attach_nullable(self):
        query = 'SELECT COUNT("IsHoliday") FROM features'
        q = QueryParser(metadata).query(query)
        assert(q._select_symbols[0].expression.xpath_first('//TableColumn').nullable == False)

        query = 'SELECT SUM(Store), "Date" as d FROM sales GROUP BY "date"'
        q = QueryParser(metadata).query(query)
        assert(q._named_symbols['d'].expression.nullable == False)
    def test_ast_attach_nullable_true(self):
        query = 'SELECT COUNT("IsHoliday") FROM sales'
        q = QueryParser(metadata).query(query)
        assert(q._select_symbols[0].expression.xpath_first('//TableColumn').nullable == True)

        query = 'SELECT SUM(Store), "Date" as d FROM features GROUP BY "date"'
        q = QueryParser(metadata).query(query)
        assert(q._named_symbols['d'].expression.nullable == True)
