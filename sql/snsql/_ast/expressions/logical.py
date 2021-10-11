from snsql._ast.tokens import *
import operator
import numpy as np
from datetime import datetime, date

ops = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "=": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    "and": np.logical_and,
    "or": np.logical_or,
}


class BooleanCompare(SqlExpr):
    """ AND, OR, <=, >=, etc """

    def __init__(self, left, op, right):
        self.left = left
        self.right = right
        self.op = op

    def symbol(self, relations):
        return BooleanCompare(self.left.symbol(relations), self.op, self.right.symbol(relations))

    def type(self):
        return "boolean"

    def sensitivity(self):
        return 1

    def children(self):
        return [self.left, self.op, self.right]

    def coerce_string(self, val, typed_val):
        # SQL-92 rules for casting types in comparison
        if isinstance(typed_val, bool):
            return parse_bool(val)
        elif isinstance(typed_val, int):
            try:
                v = int(val)
            except:
                v = float(val)
            return v
        elif isinstance(typed_val, float):
            return float(val)
        elif isinstance(typed_val, datetime):
            return datetime.fromisoformat(val)
        elif isinstance(typed_val, date):
            return date.fromisoformat(val)
        else:
            return val

    def evaluate(self, bindings):
        l = self.left.evaluate(bindings)
        r = self.right.evaluate(bindings)
        if type(l) != type(r):
            if isinstance(l, str):
                l = self.coerce_string(l, r)
            elif isinstance(r, str):
                r = self.coerce_string(r, l)
        try:
            res = bool(ops[self.op.lower()](l, r))
        except:
            raise ValueError(
                "We don't know how to compare {0} {1} {2} of mismatched types {3} and {4}".format(
                    l, self.op, r, str(type(l)), str(type(r))
                )
            )

        return parse_bool(res)


class ColumnBoolean(SqlExpr):
    """A qualified column name that was parsed in a context that requires boolean"""

    def __init__(self, expression):
        self.expression = expression

    def symbol(self, relations):
        return ColumnBoolean(self.expression.symbol(relations))

    def type(self):
        return "boolean"

    def sensitivity(self):
        return 1

    def children(self):
        return [self.expression]

    def evaluate(self, bindings):
        return parse_bool(self.expression.evaluate(bindings))


class NestedBoolean(SqlExpr):
    """A nested expression with no name"""

    def __init__(self, expression):
        self.expression = expression

    def symbol(self, relations):
        return NestedBoolean(self.expression.symbol(relations))

    def type(self):
        return self.expression.type()

    def sensitivity(self):
        return self.expression.sensitivity()

    def children(self):
        return [Token("("), self.expression, Token(")")]

    def evaluate(self, bindings):
        return parse_bool(self.expression.evaluate(bindings))


class LogicalNot(SqlExpr):
    """Negation of a boolean expression"""

    def __init__(self, expression):
        self.expression = expression

    def symbol(self, relations):
        return LogicalNot(self.expression.symbol(relations))

    def type(self):
        return "boolean"

    def sensitivity(self):
        return 1

    def children(self):
        return [Token("NOT"), self.expression]

    def evaluate(self, bindings):
        val = self.expression.evaluate(bindings)
        return not parse_bool(val)


class PredicatedExpression(SqlExpr):
    def __init__(self, expression, predicate):
        self.expression = expression
        self.predicate = predicate

    def children(self):
        return [self.expression, self.predicate]

    def symbol(self, relations):
        return PredicatedExpression(
            self.expression.symbol(relations), self.predicate.symbol(relations)
        )


class InCondition(SqlExpr):
    def __init__(self, expressions, is_not=False):
        self.expressions = expressions
        self.is_not = is_not

    def children(self):
        pre = ([Token("NOT")] if self.is_not else []) + [Token("IN"), Token("(")]
        post = [Token(")")]
        return pre + [self.expressions] + post


class BetweenCondition(SqlExpr):
    def __init__(self, lower, upper, is_not=False):
        self.lower = lower
        self.upper = upper
        self.is_not = is_not

    def children(self):
        pre = [Token("NOT")] if self.is_not else [] + [Token("BETWEEN")]
        return pre + [self.lower, Token("AND"), self.upper]


class IsCondition(SqlExpr):
    def __init__(self, value, is_not=False):
        self.value = value
        self.is_not = is_not

    def children(self):
        pre = [Token("IS")] + [Token("NOT")] if self.is_not else []
        return pre + [self.value]


class CaseExpression(SqlExpr):
    """A case expression"""

    def __init__(self, expression, when_exprs, else_expr):
        self.expression = expression
        self.when_exprs = when_exprs
        self.else_expr = else_expr

    def symbol(self, relations):
        return CaseExpression(
            self.expression.symbol(relations) if self.expression is not None else None,
            [we.symbol(relations) for we in self.when_exprs],
            self.else_expr.symbol(relations) if self.else_expr is not None else None,
        )

    def type(self):
        t = [self.else_expr.type()] if self.else_expr is not None else []
        t = t + [we.type() for we in self.when_exprs]
        if len(unique(t)) == 1:
            return t[0]
        elif "string" in t:
            return "string"
        elif sorted(unique(t)) == ["float", "int"]:
            return "float"
        else:
            return "unknown"

    def sensitivity(self):
        t = [self.else_expr.sensitivity()] if self.else_expr is not None else []
        t = t + [we.sensitivity() for we in self.when_exprs]
        t = [s for s in t if s is not None]
        if len(t) > 0:
            return max(t)
        else:
            return None

    def children(self):
        return (
            [Token("CASE"), self.expression]
            + self.when_exprs
            + ([Token("ELSE "), self.else_expr] if self.else_expr is not None else [])
            + [Token("END")]
        )

    def evaluate(self, bindings):
        if self.expression is not None:
            # simple search
            for we in self.when_exprs:
                if BooleanCompare(self.expression, Op("="), we.expression).evaluate(bindings):
                    return we.then.evaluate(bindings)
        else:
            # regular search
            for we in self.when_exprs:
                if we.expression.evaluate(bindings):
                    return we.then.evaluate(bindings)
        return (
            self.else_expr.evaluate(bindings)
            if self.else_expr
            else None
            )


class WhenExpression(SqlExpr):
    """A when expression in a case expression"""

    def __init__(self, expression, then):
        self.expression = expression
        self.then = then

    def symbol(self, relations):
        return WhenExpression(self.expression.symbol(relations), self.then.symbol(relations))

    def type(self):
        return self.then.type()

    def sensitivity(self):
        return self.then.sensitivity()

    def children(self):
        return [Token("WHEN"), self.expression, Token("THEN"), self.then]

    def evaluate(self, bindings):
        if self.expression.evaluate(bindings):
            return self.then.evaluate(bindings)
        else:
            return None


class ChooseFunction(SqlExpr):
    def __init__(self, expression, choices):
        self.expression = expression
        self.choices = choices
    def children(self):
        return [Token("CHOOSE"), Token("("), self.expression, Token(","), self.choices, Token(")")]
    def evaluate(self, bindings):
        idx = int(self.expression.evaluate(bindings))
        # index in CHOICE is 1-based, not 0-based
        if len(self.choices) < idx or idx < 1:
            return Literal(None).evaluate(bindings) # NULL
        return self.choices[idx - 1].evaluate(bindings)




class IIFFunction(SqlExpr):
    def __init__(self, test, yes, no):
        self.test = test
        self.yes = yes
        self.no = no

    def children(self):
        return [
            Token("IIF"),
            Token("("),
            self.test,
            Token(","),
            self.yes,
            Token(","),
            self.no,
            Token(")"),
        ]
    def evaluate(self, bindings):
        if (self.test.evaluate(bindings) == True):
            return self.yes.evaluate(bindings)
        return self.no.evaluate(bindings)

def parse_bool(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, (int, float)):
        if float(v) == 0.0:
            return False
        elif float(v) == 1.0:
            return True
        else:
            return v
    elif isinstance(v, str):
        if v.lower() == "true" or v == "1":
            return True
        elif v.lower() == "false" or v == "0":
            return False
        else:
            return v
