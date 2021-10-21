import datetime
from snsql.sql.parse import QueryParser
import calendar

class TestDateTime:
    def test_cur_date(self):
        frag = "CURRENT_DATE"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(isinstance(v, datetime.date))
    def test_cur_time(self):
        frag = "CURRENT_TIME"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(isinstance(v, datetime.time))
    def test_cur_timestamp(self):
        frag = "CURRENT_TIMESTAMP"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(isinstance(v, datetime.datetime))
    def test_extract_0(self):
        frag = "EXTRACT(SECOND FROM '2017-05-20')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 0)
    def test_extract_null(self):
        frag = "EXTRACT(DAY FROM '09:15:07')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v is None)
    def test_extract_second(self):
        frag = "EXTRACT(SECOND FROM '2017-05-20 09:15:07')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 7)
    def test_extract_day(self):
        frag = "EXTRACT(DAY FROM '2017-05-20')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 20)
    def test_extract_year(self):
        frag = "EXTRACT(YEAR FROM CURRENT_DATE)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v > 2020)
    def test_extract_month(self):
        frag = "EXTRACT(MONTH FROM '2017-05-20')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 5)
    def test_extract_seconds(self):
        frag = "EXTRACT(SECOND FROM '09:15:07')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 7)
    def test_extract_minutes(self):
        frag = "EXTRACT(MINUTE FROM '09:15:07')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 15)
    def test_extract_hours(self):
        frag = "EXTRACT(HOUR FROM '09:15:07')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 9)
    def test_extract_weekday(self):
        frag = "EXTRACT(WEEKDAY FROM '2017-05-20')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 5)
    def test_extract_micro_date(self):
        frag = "EXTRACT(MICROSECOND FROM '2017-05-20')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 0)
    def test_extract_micro_time_no_micro(self):
        frag = "EXTRACT(MICROSECOND FROM '09:12:32')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 0)
    def test_extract_micro(self):
        frag = "EXTRACT(MICROSECOND FROM CURRENT_TIMESTAMP)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v > 0)
    def test_day_name(self):
        frag = "DAYNAME('2017-05-20 09:15:07')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == calendar.day_name[5])
