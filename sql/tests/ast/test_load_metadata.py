from snsql.metadata import Metadata
from snsql.sql.parse import QueryParser

def load_metadata():
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
        "": {
            "": {
                "features": table_features,
                "sales": table_sales,
                }
            },
        }
    return Metadata.from_dict(metadata_dict)
schema = load_metadata()
#   Unit tests
#
class TestLoadMeta:
    def test_join_query(self):
        query = 'SELECT COUNT(Store), COUNT(*) FROM sales'
        _ = QueryParser(schema).query(query)
        query = 'SELECT COUNT(table1.Store), COUNT(*) FROM sales AS table1'
        _ = QueryParser(schema).query(query)
        query = 'SELECT COUNT(sales.Store), COUNT(*) FROM sales'
        _ = QueryParser(schema).query(query)
    def test_join_query(self):
        query = 'SELECT COUNT(sales.Store) FROM sales JOIN features ON sales.Store = features.Store'
        _ = QueryParser(schema).query(query)
        query = 'SELECT COUNT(table1.Store) FROM sales AS table1 JOIN features ON table1.Store = features.Store'
        _ = QueryParser(schema).query(query)
    def test_subqueries_query(self):
        query = 'SELECT SUM(subquery.Store), SUM(avg_price) FROM (SELECT Store, Temperature, AVG(table1.Fuel_Price) AS avg_price FROM features AS table1 GROUP BY Store, Temperature) AS subquery GROUP BY Temperature;'
        _ = QueryParser(schema).query(query)
        query = 'SELECT SUM(Store), SUM(avg_price) FROM (SELECT Store, Temperature, AVG(table1.Fuel_Price) AS avg_price FROM features AS table1 GROUP BY Store, Temperature) AS subquery GROUP BY Temperature;'
        _ = QueryParser(schema).query(query)
        query = 'SELECT SUM(avg_price) FROM (SELECT AVG(Fuel_Price) AS avg_price FROM features GROUP BY IsHoliday) AS subquery;'
        _ = QueryParser(schema).query(query)

