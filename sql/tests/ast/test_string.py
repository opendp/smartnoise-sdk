from snsql._ast.expressions.string import *
from snsql._ast.tokens import Literal
from snsql.sql.parse import QueryParser

class TestStringFunctions:
    def test_concat_null(self):
        frag = "CONCAT('q', NULL, 'abc')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v is None)
    def test_concat_1(self):
        frag = "CONCAT(15 * 3)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == '45')
    def test_concat_3(self):
        frag = "CONCAT(CONCAT('a', 'b'), 15 * 3, 'z')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 'ab45z')
    def test_coalesce(self):
        frag = "COALESCE(NULL, UPPER('xyz'), NULL, 'abz')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 'XYZ')
    def test_coalesce_null(self):
        frag = "COALESCE(NULL, NULL)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == None)
    def test_lower_upper(self):
        frag = "LOWER(COALESCE(NULL, UPPER('xYz'), NULL, 'abz'))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 'xyz')
    def test_upper_lower(self):
        frag = "UPPER(COALESCE(NULL, LOWER('xYz'), NULL, 'abz'))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 'XYZ')
    def test_trim(self):
        frag = "TRIM('   xYz   ')"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 'xYz')
    def test_trim_empty(self):
        frag = "TRIM(COALESCE('\t', '   xYz   ', ' '))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == '')
    def test_trim_NULL(self):
        frag = "TRIM(COALESCE(NULL, NULL))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v is None)
    def test_char_length(self):
        frag = "CHAR_LENGTH(TRIM('   xYz   '))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 3)
    def test_position_not_in(self):
        frag = "POSITION('q' IN TRIM('   xYz   '))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 0)
    def test_position_null(self):
        frag = "POSITION('q' IN COALESCE(NULL, NULL))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == None)
    def test_position(self):
        frag = "POSITION('z' IN TRIM('   xYz   '))"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 3)
    def test_position_neg(self):
        frag = "SUBSTRING('Hello world!' FROM -4 FOR 2)"
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 'rl')
