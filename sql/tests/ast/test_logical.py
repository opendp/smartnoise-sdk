import pytest
from snsql._ast.expressions.logical import *
from snsql._ast.tokens import Column, Literal
from snsql.sql.parse import QueryParser
from datetime import date
import numpy as np

"""
    Test evaluation of logical operators.  We need to support
    operands of various types, including:
        * Parsed atomic types
        * Atomic types loaded from the database
        * String literal representations of types
        * Column names bound to a column value in the row

    Test harness creates arrays with several representations of
    values for numeric, date, and string
"""

vals_5 = [5, 5.0, "5", "5.0", float(5.0), int(5)]
names_5 = ["i5", "f5", "si5", "sf5", "npf5", "npi5"]

vals_7 = [7, 7.0, "7", "7.0"]
names_7 = ["i7", "f7", "si7", "sf7"]

d1_str = "1978-06-30"
d2_str = "1984-10-14"

vals_d1 = [d1_str, date.fromisoformat(d1_str)]
names_d1 = ["sd1", "dd1"]

vals_d2 = [d2_str, date.fromisoformat(d2_str)]
names_d2 = ["sd2", "dd2"]

vals_str = ["Smart", "Noise"]
names_str = ["smart", "noise"]

vals_f = ["false", False]
names_f = ["s_f",  "b_f"]

vals_t = ["true", True]
names_t = ["s_t", "b_t"]

# now load all the values into a bindings dict
vals = vals_5 + vals_7 + vals_d1 + vals_d2 + vals_str + vals_t + vals_f
names = names_5 + names_7 + names_d1 + names_d2 + names_str + names_t + names_f
bindings = dict((name.lower(), val ) for name, val in zip(names, vals))

class TestLogical:
    def test_eq(Self):
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            # All True
            for v5b, n5b in zip(vals_5, names_5):
                if not (isinstance(v5, str) and isinstance(v5b, str)):
                    assert(BooleanCompare(Literal(v5), '=', Literal(v5b)).evaluate(None))
                    assert(BooleanCompare(Column(n5), '=', Literal(v5b)).evaluate(bindings))
                    assert(BooleanCompare(Literal(v5), '=', Column(n5b)).evaluate(bindings))
            # All False
            for v7, n7 in zip(vals_7, names_7):
                if not (isinstance(v5, str) and isinstance(v7, str)):
                    assert(not BooleanCompare(Literal(v5), '=', Literal(v7)).evaluate(None))
                    assert(not BooleanCompare(Column(n5), '=', Literal(v7)).evaluate(bindings))
                    assert(not BooleanCompare(Literal(v5), '=', Column(n7)).evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d1b, n1b in zip(vals_d1, names_d1):
                # all True
                comp = BooleanCompare(Literal(d1b), '=', Literal(d1))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n1b), '=', Literal(d1))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1b), '=', Column(n1))
                assert(comp.evaluate(bindings))
            for d2, n2 in zip(vals_d2, names_d2):
                # all False
                comp = BooleanCompare(Literal(d1), '=', Literal(d2))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n1), '=', Literal(d2))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1), '=', Column(n2))
                assert(not comp.evaluate(bindings))
        # test strings
        # All True
        assert(BooleanCompare(Literal(vals_str[0]), '=', Literal(vals_str[0])).evaluate(None))
        assert(BooleanCompare(Column(names_str[0]), '=', Literal(vals_str[0])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[0]), '=', Column(names_str[0])).evaluate(bindings))
        # All False
        assert(not BooleanCompare(Literal(vals_str[1]), '=', Literal(vals_str[0])).evaluate(None))
        assert(not BooleanCompare(Column(names_str[1]), '=', Literal(vals_str[0])).evaluate(bindings))
        assert(not BooleanCompare(Literal(vals_str[1]), '=', Column(names_str[0])).evaluate(bindings))
    def test_neq(Self):
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            # All False
            for v5b, n5b in zip(vals_5, names_5):
                if not (isinstance(v5, str) and isinstance(v5b, str)):
                    assert(not BooleanCompare(Literal(v5), '!=', Literal(v5b)).evaluate(None))
                    assert(not BooleanCompare(Column(n5), '!=', Literal(v5b)).evaluate(bindings))
                    assert(not BooleanCompare(Literal(v5), '!=', Column(n5b)).evaluate(bindings))
            # All True
            for v7, n7 in zip(vals_7, names_7):
                if not (isinstance(v5, str) and isinstance(v7, str)):
                    assert(BooleanCompare(Literal(v5), '<>', Literal(v7)).evaluate(None))
                    assert(BooleanCompare(Column(n5), '!=', Literal(v7)).evaluate(bindings))
                    assert(BooleanCompare(Literal(v5), '<>', Column(n7)).evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d1b, n1b in zip(vals_d1, names_d1):
                # all False
                comp = BooleanCompare(Literal(d1b), '!=', Literal(d1))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n1b), '<>', Literal(d1))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1b), '!=', Column(n1))
                assert(not comp.evaluate(bindings))
            for d2, n2 in zip(vals_d2, names_d2):
                # all True
                comp = BooleanCompare(Literal(d1), '!=', Literal(d2))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n1), '<>', Literal(d2))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1), '!=', Column(n2))
                assert(comp.evaluate(bindings))
        # test strings
        # All False
        assert(not BooleanCompare(Literal(vals_str[0]), '!=', Literal(vals_str[0])).evaluate(None))
        assert(not BooleanCompare(Column(names_str[0]), '<>', Literal(vals_str[0])).evaluate(bindings))
        assert(not BooleanCompare(Literal(vals_str[0]), '!=', Column(names_str[0])).evaluate(bindings))
        # All True
        assert(BooleanCompare(Literal(vals_str[1]), '!=', Literal(vals_str[0])).evaluate(None))
        assert(BooleanCompare(Column(names_str[1]), '<>', Literal(vals_str[0])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[1]), '!=', Column(names_str[0])).evaluate(bindings))

    def test_gt(self):
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            for v7, n7 in zip(vals_7, names_7):
                # all True
                comp = BooleanCompare(Literal(v7), '>', Literal(v5))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n7), '>', Literal(v5))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v7), '>', Column(n5))
                assert(comp.evaluate(bindings))
                # all False
                comp = BooleanCompare(Literal(v5), '>', Literal(v7))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n5), '>', Literal(v7))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v5), '>', Column(n7))
                assert(not comp.evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d2, n2 in zip(vals_d2, names_d2):
                # all True
                comp = BooleanCompare(Literal(d2), '>', Literal(d1))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n2), '>', Literal(d1))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d2), '>', Column(n1))
                assert(comp.evaluate(bindings))
                # all False
                comp = BooleanCompare(Literal(d1), '>', Literal(d2))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n1), '>', Literal(d2))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1), '>', Column(n2))
                assert(not comp.evaluate(bindings))
        # test strings
        # All True
        assert(BooleanCompare(Literal(vals_str[0]), '>', Literal(vals_str[1])).evaluate(None))
        assert(BooleanCompare(Column(names_str[0]), '>', Literal(vals_str[1])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[0]), '>', Column(names_str[1])).evaluate(bindings))
        # All False
        assert(not BooleanCompare(Literal(vals_str[1]), '>', Literal(vals_str[0])).evaluate(None))
        assert(not BooleanCompare(Column(names_str[1]), '>', Literal(vals_str[0])).evaluate(bindings))
        assert(not BooleanCompare(Literal(vals_str[1]), '>', Column(names_str[0])).evaluate(bindings))
    def test_lt(self):
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            for v7, n7 in zip(vals_7, names_7):
                # all False
                comp = BooleanCompare(Literal(v7), '<', Literal(v5))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n7), '<', Literal(v5))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v7), '<', Column(n5))
                assert(not comp.evaluate(bindings))
                # all True
                comp = BooleanCompare(Literal(v5), '<', Literal(v7))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n5), '<', Literal(v7))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v5), '<', Column(n7))
                assert(comp.evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d2, n2 in zip(vals_d2, names_d2):
                # all False
                comp = BooleanCompare(Literal(d2), '<', Literal(d1))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n2), '<', Literal(d1))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d2), '<', Column(n1))
                assert(not comp.evaluate(bindings))
                # all True
                comp = BooleanCompare(Literal(d1), '<', Literal(d2))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n1), '<', Literal(d2))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1), '<', Column(n2))
                assert(comp.evaluate(bindings))
        # test strings
        # All False
        assert(not BooleanCompare(Literal(vals_str[0]), '<', Literal(vals_str[1])).evaluate(None))
        assert(not BooleanCompare(Column(names_str[0]), '<', Literal(vals_str[1])).evaluate(bindings))
        assert(not BooleanCompare(Literal(vals_str[0]), '<', Column(names_str[1])).evaluate(bindings))
        # All True
        assert(BooleanCompare(Literal(vals_str[1]), '<', Literal(vals_str[0])).evaluate(None))
        assert(BooleanCompare(Column(names_str[1]), '<', Literal(vals_str[0])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[1]), '<', Column(names_str[0])).evaluate(bindings))
    def test_gte(self):
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            for v7, n7 in zip(vals_7, names_7):
                # all True
                comp = BooleanCompare(Literal(v7), '>=', Literal(v5))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n7), '>=', Literal(v5))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v7), '>=', Column(n5))
                assert(comp.evaluate(bindings))
                # all False
                comp = BooleanCompare(Literal(v5), '>=', Literal(v7))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n5), '>=', Literal(v7))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v5), '>=', Column(n7))
                assert(not comp.evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d2, n2 in zip(vals_d2, names_d2):
                # all True
                comp = BooleanCompare(Literal(d2), '>=', Literal(d1))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n2), '>=', Literal(d1))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d2), '>=', Column(n1))
                assert(comp.evaluate(bindings))
                # all False
                comp = BooleanCompare(Literal(d1), '>=', Literal(d2))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n1), '>=', Literal(d2))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1), '>=', Column(n2))
                assert(not comp.evaluate(bindings))
        # test strings
        # All True
        assert(BooleanCompare(Literal(vals_str[0]), '>=', Literal(vals_str[1])).evaluate(None))
        assert(BooleanCompare(Column(names_str[0]), '>=', Literal(vals_str[1])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[0]), '>=', Column(names_str[1])).evaluate(bindings))
        # All False
        assert(not BooleanCompare(Literal(vals_str[1]), '>=', Literal(vals_str[0])).evaluate(None))
        assert(not BooleanCompare(Column(names_str[1]), '>=', Literal(vals_str[0])).evaluate(bindings))
        assert(not BooleanCompare(Literal(vals_str[1]), '>=', Column(names_str[0])).evaluate(bindings))
        # Test equality
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            # All True
            for v5b, n5b in zip(vals_5, names_5):
                if not (isinstance(v5, str) and isinstance(v5b, str)):
                    assert(BooleanCompare(Literal(v5), '>=', Literal(v5b)).evaluate(None))
                    assert(BooleanCompare(Column(n5), '>=', Literal(v5b)).evaluate(bindings))
                    assert(BooleanCompare(Literal(v5), '>=', Column(n5b)).evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d1b, n1b in zip(vals_d1, names_d1):
                # all True
                comp = BooleanCompare(Literal(d1b), '>=', Literal(d1))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n1b), '>=', Literal(d1))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1b), '>=', Column(n1))
                assert(comp.evaluate(bindings))
        # test strings
        # All True
        assert(BooleanCompare(Literal(vals_str[0]), '>=', Literal(vals_str[0])).evaluate(None))
        assert(BooleanCompare(Column(names_str[0]), '>=', Literal(vals_str[0])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[0]), '>=', Column(names_str[0])).evaluate(bindings))

    def test_lte(self):
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            for v7, n7 in zip(vals_7, names_7):
                # all False
                comp = BooleanCompare(Literal(v7), '<=', Literal(v5))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n7), '<=', Literal(v5))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v7), '<=', Column(n5))
                assert(not comp.evaluate(bindings))
                # all True
                comp = BooleanCompare(Literal(v5), '<=', Literal(v7))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n5), '<=', Literal(v7))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(v5), '<=', Column(n7))
                assert(comp.evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d2, n2 in zip(vals_d2, names_d2):
                # all False
                comp = BooleanCompare(Literal(d2), '<=', Literal(d1))
                assert(not comp.evaluate(None))
                comp = BooleanCompare(Column(n2), '<=', Literal(d1))
                assert(not comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d2), '<=', Column(n1))
                assert(not comp.evaluate(bindings))
                # all True
                comp = BooleanCompare(Literal(d1), '<=', Literal(d2))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n1), '<=', Literal(d2))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1), '<=', Column(n2))
                assert(comp.evaluate(bindings))
        # test strings
        # All False
        assert(not BooleanCompare(Literal(vals_str[0]), '<=', Literal(vals_str[1])).evaluate(None))
        assert(not BooleanCompare(Column(names_str[0]), '<=', Literal(vals_str[1])).evaluate(bindings))
        assert(not BooleanCompare(Literal(vals_str[0]), '<=', Column(names_str[1])).evaluate(bindings))
        # All True
        assert(BooleanCompare(Literal(vals_str[1]), '<=', Literal(vals_str[0])).evaluate(None))
        assert(BooleanCompare(Column(names_str[1]), '<=', Literal(vals_str[0])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[1]), '<=', Column(names_str[0])).evaluate(bindings))
        # Test equality
        # test numeric values
        for v5, n5 in zip(vals_5, names_5):
            # All True
            for v5b, n5b in zip(vals_5, names_5):
                if not (isinstance(v5, str) and isinstance(v5b, str)):
                    assert(BooleanCompare(Literal(v5), '<=', Literal(v5b)).evaluate(None))
                    assert(BooleanCompare(Column(n5), '<=', Literal(v5b)).evaluate(bindings))
                    assert(BooleanCompare(Literal(v5), '<=', Column(n5b)).evaluate(bindings))
        # test dates
        for d1, n1 in zip(vals_d1, names_d1):
            for d1b, n1b in zip(vals_d1, names_d1):
                # all True
                comp = BooleanCompare(Literal(d1b), '<=', Literal(d1))
                assert(comp.evaluate(None))
                comp = BooleanCompare(Column(n1b), '<=', Literal(d1))
                assert(comp.evaluate(bindings))
                comp = BooleanCompare(Literal(d1b), '<=', Column(n1))
                assert(comp.evaluate(bindings))
        # test strings
        # All True
        assert(BooleanCompare(Literal(vals_str[0]), '<=', Literal(vals_str[0])).evaluate(None))
        assert(BooleanCompare(Column(names_str[0]), '<=', Literal(vals_str[0])).evaluate(bindings))
        assert(BooleanCompare(Literal(vals_str[0]), '<=', Column(names_str[0])).evaluate(bindings))
    def test_and(self):
        for tv, tn in zip(vals_t, names_t):
            for tvb, tnb in zip(vals_t, names_t):
                # All True
                if not (isinstance(tv, str) and isinstance(tvb, str)):
                    assert BooleanCompare(Literal(tv), 'and', Literal(tvb)).evaluate(None)
                    assert BooleanCompare(Column(tn), 'and', Literal(tvb)).evaluate(bindings)
                    assert BooleanCompare(Literal(tv), 'and', Column(tnb)).evaluate(bindings)
            for fv, fn in zip(vals_f, names_f):
                # All False
                if not (isinstance(tv, str) and isinstance(fv, str)):
                    assert not BooleanCompare(Literal(tv), 'and', Literal(fv)).evaluate(None)
                    assert not BooleanCompare(Column(tn), 'and', Literal(fv)).evaluate(bindings)
                    assert not BooleanCompare(Literal(tv), 'and', Column(fn)).evaluate(bindings)

class TestCaseExpression:
    def test_simple_case(self):
        qp = QueryParser()
        c = qp.parse_expression("CASE x WHEN 5 THEN 'five' WHEN 6 THEN 'six' ELSE '' END")
        bindings = dict([('x', 5)])
        assert(c.evaluate(bindings) == "five")
        bindings = dict([('x', 6)])
        assert(c.evaluate(bindings) == 'six')
        bindings = dict([('x', 7)])
        assert(c.evaluate(bindings) == '')
    def test_variable_replace(self):
        qp = QueryParser()
        c = qp.parse_expression("CASE x WHEN 5 THEN y WHEN 6 THEN z ELSE 0 END")
        bindings = dict([('x', 5), ('y', 10), ('z', 12)])
        assert(c.evaluate(bindings) == 10)
        bindings['x'] = 6
        assert(c.evaluate(bindings) == 12)
        bindings['x'] = 1
        assert(c.evaluate(bindings) == 0)
    def test_string_bound(self):
        qp = QueryParser()
        c = qp.parse_expression("CASE x WHEN 5 THEN y WHEN 6 THEN z ELSE q END")
        bindings = dict([('x', 5), ('y', 'ten'), ('z', 'twelve'), ('q', 'zero')])
        assert(c.evaluate(bindings) == "ten")
        bindings['x'] = 6
        assert(c.evaluate(bindings) == "twelve")
        bindings['x'] = 1
        assert(c.evaluate(bindings) == "zero")
    def test_full_case(self):
        qp = QueryParser()
        c = qp.parse_expression("CASE WHEN x <= 5 THEN y WHEN x > 6 THEN 0 ELSE z END")
        bindings = dict([('x', 5), ('y', 10), ('z', 12)])
        assert(c.evaluate(bindings) == 10)
        bindings['x'] = 6
        assert(c.evaluate(bindings) == 12)
        bindings['x'] = 10
        assert(c.evaluate(bindings) == 0)
    def test_iif(self):
        qp = QueryParser()
        c = qp.parse_expression("IIF(x <= 5, y, 0)")
        bindings = dict([('x', 5), ('y', 10), ('z', 12)])
        assert(c.evaluate(bindings) == 10)
        bindings["x"] = 6
        assert(c.evaluate(bindings) == 0)
        c = qp.parse_expression("IIF(x <= 5, y, 'string')")
        assert(c.evaluate(bindings) == "string")
    def test_choose(self):
        qp = QueryParser()
        c = qp.parse_expression("CHOOSE(x, 'a', 'b', 'c')")
        bindings = dict([('x', 3), ('y', 10), ('z', 12)])
        assert(c.evaluate(bindings) == "c")
        bindings["x"] = 1
        assert(c.evaluate(bindings) == 'a')
        bindings["x"] = 0
        assert(c.evaluate(bindings) == None)
        bindings["x"] = 10
        assert(c.evaluate(bindings) == None)
        c = qp.parse_expression("CHOOSE(x, 'a', 5, NULL)")
        bindings = dict([('x', 3), ('y', 10), ('z', 12)])
        assert(c.evaluate(bindings) == None)
        bindings["x"] = "2"
        assert(c.evaluate(bindings) == 5)
        c = qp.parse_expression("CHOOSE(x % 2 + 1, NULL, 5)")
        bindings["x"] = 13
        assert(c.evaluate(bindings) == 5)

class TestPredicateExpression:
    def test_between_condition(self):
        c = PredicatedExpression(Column("x"), BetweenCondition(Literal(3), Literal(6), False))
        bindings = dict([('x', 2)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', 4)])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', 7)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), BetweenCondition(Literal('d'), Literal('h'), False))
        bindings = dict([('x', 'a')])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', 'e')])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', 'v')])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), BetweenCondition(Literal('2017/01/01'), Literal('2019/01/01'), False))
        bindings = dict([('x', '2016/01/01')])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', '2018/01/01')])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', '2020/01/01')])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

    def test_not_between_condition(self):
        c = PredicatedExpression(Column("x"), BetweenCondition(Literal(3), Literal(6), True))
        bindings = dict([('x', 2)])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', 4)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', 7)])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), BetweenCondition(Literal('d'), Literal('h'), True))
        bindings = dict([('x', 'a')])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', 'e')])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', 'v')])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), BetweenCondition(Literal('2017/01/01'), Literal('2019/01/01'), True))
        bindings = dict([('x', '2016/01/01')])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', '2018/01/01')])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', '2020/01/01')])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

    def test_in_condition(self):
        c = PredicatedExpression(Column("x"), InCondition(Seq([Literal(1), Literal(2), Literal(3)])))
        bindings = dict([('x', 2)])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', 2.)])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', 7)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), InCondition(Seq([Literal("1"), Literal("2"), Literal("3")])))
        bindings = dict([('x', "2")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "2.")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "7")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), InCondition(Seq([Literal("2017/01/01"), Literal("2019/01/01")])))
        bindings = dict([('x', "2017/01/01")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "2020/01/01")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

    def test_not_in_condition(self):
        c = PredicatedExpression(Column("x"), InCondition(Seq([Literal(1), Literal(2), Literal(3)]),True))
        bindings = dict([('x', 2)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', 2.)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', 7)])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), InCondition(Seq([Literal("1"), Literal("2"), Literal("3")]),True))
        bindings = dict([('x', "2")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "2.")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "7")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

        c = PredicatedExpression(Column("x"), InCondition(Seq([Literal("2017/01/01"), Literal("2019/01/01")]),True))
        bindings = dict([('x', "2017/01/01")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "2020/01/01")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == None)

    def test_is_condition(self):
        c = PredicatedExpression(Column("x"), IsCondition(Literal("NULL"), False))
        bindings = dict([('x', 2)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == True)

        c = PredicatedExpression(Column("x"), IsCondition(Literal(None), False))
        bindings = dict([('x', 2)])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == True)

        c = PredicatedExpression(Column("x"), IsCondition(Literal("TRUE"), False))
        bindings = dict([('x', "True")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "true")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "t")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "y")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "yes")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "on")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "1")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == False)

        c = PredicatedExpression(Column("x"), IsCondition(Literal("False"), False))
        bindings = dict([('x', "False")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "false")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "f")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "n")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "no")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "off")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', "0")])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == False)

    def test_is_not_condition(self):
        c = PredicatedExpression(Column("x"), IsCondition(Literal("NULL"), True))
        bindings = dict([('x', 2)])
        assert(c.evaluate(bindings) == True)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == False)

        c = PredicatedExpression(Column("x"), IsCondition(Literal("True"), True))
        bindings = dict([('x', "True")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "true")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "t")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "y")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "yes")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "on")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "1")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == True)

        c = PredicatedExpression(Column("x"), IsCondition(Literal("False"), True))
        bindings = dict([('x', "False")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "false")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "f")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "n")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "no")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "off")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', "0")])
        assert(c.evaluate(bindings) == False)
        bindings = dict([('x', None)])
        assert(c.evaluate(bindings) == True)
