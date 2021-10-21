from snsql._ast.expressions.numeric import TruncFunction
from snsql._ast.tokens import Literal
from snsql.sql.parse import QueryParser

class TestNumericFunctions:
    def test_truncate_float(self):
        frag = '15.3333333333333 * 5'
        expr = QueryParser().parse_expression(frag)
        s = TruncFunction(expr, Literal(2))
        v = s.evaluate({})
        assert(v == 76.66)
    def test_truncate_int(self):
        frag = '15 * 5'
        expr = QueryParser().parse_expression(frag)
        s = TruncFunction(expr, Literal(-1))
        v = s.evaluate({})
        assert(v == 70)
    def test_truncate_parse(self):
        frag = 'TRUNCATE(15 * 5, -1)'
        expr = QueryParser().parse_expression(frag)
        assert(frag.replace(' ', '') == str(expr).replace(' ', ''))
        v = expr.evaluate({})
        assert(v == 70)
