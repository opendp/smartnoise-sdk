from opendp.whitenoise.ast.tokens import *

import numpy as np

ops = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    'and': np.logical_and,
    'or': np.logical_or
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
    def evaluate(self, bindings):
        l = self.left.evaluate(bindings)
        r = self.right.evaluate(bindings)
        return ops[self.op.lower()](l, r)

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
        return self.expression.evaluate(bindings)

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
        return np.logical_not(self.expression.evaluate(bindings))

class PredicatedExpression(SqlExpr):
    def __init__(self, expression, predicate):
        self.expression = expression
        self.predicate = predicate
    def children(self):
        return [self.expression, self.predicate]
    def symbol(self, relations):
        return PredicatedExpression(self.expression.symbol(relations), self.predicate.symbol(relations))

class InCondition(SqlExpr):
    def __init__(self, expressions, is_not=False):
        self.expressions = expressions
        self.is_not = is_not
    def children(self):
        pre = ([Token("NOT")] if self.is_not else []) + [Token("IN"), Token('(')]
        post = [Token(')')]
        return pre + [self.expressions] + post


class BetweenCondition(SqlExpr):
    def __init__(self, lower, upper, is_not=False):
        self.lower = lower
        self.upper = upper
        self.is_not = is_not
    def children(self):
        pre = [Token("NOT")] if self.is_not else [] + [Token("BETWEEN")]
        return pre + [self.lower, Token('AND'), self.upper]

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
        return CaseExpression( \
            self.expression.symbol(relations) if self.expression is not None else None,\
            [we.symbol(relations) for we in self.when_exprs], \
            self.else_expr.symbol(relations) if self.else_expr is not None else None)
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
        return [Token("CASE"), self.expression] + self.when_exprs + ([Token("ELSE "), self.else_expr] if self.else_expr is not None else []) + [Token("END")]
    def evaluate(self, bindings):
        else_exp = self.else_expr.evaluate(bindings)
        res = np.repeat(else_exp, len(bindings[list(bindings.keys())[0]]))
        if self.expression is not None:
            # simple search
            for we in self.when_exprs:
                match = BooleanCompare(self.expression, '=', we.expression).evaluate(bindings)
                res[match] = we.then.evaluate(bindings)
        else:
            # regular search
            for we in self.when_exprs:
                match = we.expression.evaluate(bindings)
                res[match] = we.then.evaluate(bindings)
        return res

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
        return [Token('CHOOSE'), Token('('), self.expression, Token(','), self.choices, Token(')')]


class IIFFunction(SqlExpr):
    def __init__(self, test, yes, no):
        self.test = test
        self.yes = yes
        self.no = no
    def children(self):
        return [Token('IIF'), Token('('), self.test, Token(','), self.yes, Token(','), self.no, Token(')') ]
