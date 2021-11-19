from snsql.metadata import Metadata
from snsql.sql.parse import QueryParser

table_features = {
    "Store": {"name": "Store", "type": "int", "private_id": True},
    "Date": {"name": "Date", "type": "datetime"},
    "Temperature": {"name": "Temperature", "type": "float"},
    "Fuel_Price": {"name": "Fuel_Price", "type": "float"},
    "IsHoliday": {"name": "IsHoliday", "type": "boolean"},
    "row_privacy": False,
}

table_sales = {
    "Store": {"name": "Store", "type": "int", "private_id": True},
    "Date": {"name": "Date", "type": "datetime"},
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

class TestLoadSymbols:
    def test_group_column_shared(self):
        query = 'SELECT "IsHoliday" As col, SUM(features."Store") FROM features GROUP BY col'
        q = QueryParser(metadata).query(query)
        assert(q._named_symbols['col'].expression == q._grouping_symbols[0].expression)

    def test_two_tables(self):
        query = 'SELECT "Temperature" AS temp, AVG(Weekly_Sales) AS sales FROM features, sales'
        q = QueryParser(metadata).query(query)
        assert(q._named_symbols['temp'].expression.tablename == 'features')
        assert(q._named_symbols['sales'].expression.xpath_first('//TableColumn').tablename == 'sales')

    def test_same_colname(self):
        query = 'SELECT sales."Store", features."Store" FROM sales, features'
        q = QueryParser(metadata).query(query)
        assert(q._named_symbols['"sales_Store"'].expression.tablename == 'sales')
        assert(q._named_symbols['"sales_Store"'].expression.colname == 'Store')
        assert(q._named_symbols['"features_Store"'].expression.tablename == 'features')
        assert(q._named_symbols['"features_Store"'].expression.colname == 'Store')
