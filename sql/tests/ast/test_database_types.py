from snsql._ast.expressions.string import *
from snsql.sql.parse import QueryParser
import datetime


class TestTypes:
    def test_cast_char(self):
        frag = "CAST(4.0 / 3.0 AS CHAR(3))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == '1.3')
    def test_cast_varchar(self):
        frag = "CAST(1.0 / 2.0 AS VARCHAR)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == '0.5')
    def test_cast_float(self):
        frag = "CAST(EXTRACT(WEEKDAY FROM CAST('2017-05-10 09:01:01' AS TIMESTAMP)) AS FLOAT)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(isinstance(v, float))
        assert(v == 2.0)
    def test_cast_time(self):
        frag = "CAST('09:05:05' AS TIME)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(isinstance(v, datetime.time))
    def test_cast_date(self):
        frag = "CAST('2017-05-10' AS DATE)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(isinstance(v, datetime.date))

    def test_cast_int(self):
        frag = "CAST(SUBSTRING('S3' FROM 2) AS INTEGER)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 3)
