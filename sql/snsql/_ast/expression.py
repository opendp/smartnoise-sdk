from .tokens import *

from .expressions.numeric import *
from .expressions.logical import *
from .expressions.sql import *


EXPR_TYPE = Union[
    "Expression",
    "Column",
    "ArithmeticExpression",
    "CaseExpression",
    "AllColumns",
    "Literal",
    "RankingFunction",
    "BareFunction",
    "RoundFunction",
    "PowerFunction",
    "AggFunction",
    "MathFunction",
    "IIFFunction",
    "ChooseFunction",
    "AliasedSubquery",
    "NestedExpression",
]

class Expression(SqlExpr):
    """A bare expression with no name"""

    def __init__(self, fragment):
        self.fragment = fragment

    def __str__(self):
        return self.fragment

    def __eq__(self, other):
        return type(self) == type(other) and self.fragment == other.fragment

    def __hash__(self):
        return hash(self.fragment)

    def symbol(self, relations):
        raise ValueError("Cannot load symbol on bare expression: " + str(self))


class NestedExpression(SqlExpr):
    """A nested expression with no name"""

    def __init__(self, expression):
        self.expression = expression

    def symbol(self, relations):
        return NestedExpression(self.expression.symbol(relations))

    def type(self):
        return self.expression.type()

    def sensitivity(self):
        return self.expression.sensitivity()

    def children(self):
        return [Token("("), self.expression, Token(")")]

    def evaluate(self, bindings):
        return self.expression.evaluate(bindings)

    @property
    def is_key_count(self):
        return self.expression.is_key_count

    @property
    def is_count(self):
        return self.expression.is_count


class NamedExpression(SqlExpr):
    """An expression with optional name"""

    def __init__(
        self, name: Identifier, expression: EXPR_TYPE
    ) -> None:
        self.name = name
        self.expression = expression

    def column_name(self):
        if self.name is not None:
            return self.name
        elif type(self.expression) is Column:
            parts = self.expression.name.split(".")
            return parts[0] if len(parts) == 1 else parts[1]
        elif type(self.expression) is AllColumns:
            return "???"
        else:
            return "???"

    def type(self):
        return self.expression.type()

    def sensitivity(self):
        return self.expression.sensitivity()

    def children(self):
        return [self.expression] + ([Token("AS"), self.name] if self.name is not None else [])

    def evaluate(self, bindings):
        return self.expression.evaluate(bindings)

    @property
    def is_key_count(self):
        return self.expression.is_key_count

    @property
    def is_count(self):
        return self.expression.is_count
