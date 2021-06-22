from .tokens import *

from .expressions.numeric import *
from .expressions.logical import *
from .expressions.sql import *

ExpressionType = Union[
    Column,
    CaseExpression,
    AllColumns,
    ArithmeticExpression,
    Literal,
    AggFunction,
    'AliasedSubquery',
    'NestedExpression',
    IIFFunction,
    RoundFunction,
    MathFunction,
    ChooseFunction,
    PowerFunction,
    BareFunction,
    RankingFunction,
]

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

    def __init__(self, fragment: str):
        self.fragment = fragment

    def __str__(self) -> str:
        return self.fragment

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.fragment == other.fragment

    def __hash__(self) -> int:
        return hash(self.fragment)

    def symbol(self, relations): #type: ignore
        raise ValueError("Cannot load symbol on bare expression: " + str(self))


class NestedExpression(SqlExpr):
    """A nested expression with no name"""

    def __init__(self, expression: ExpressionType):
        self.expression = expression

    def symbol(self, relations) -> 'NestedExpression': # type: ignore
        return NestedExpression(self.expression.symbol(relations))

    def type(self) -> str:
        return self.expression.type()

    def sensitivity(self) -> Optional[Union[int, float]]:
        return self.expression.sensitivity()

    def children(self) -> List[Union[Token, ExpressionType]]:
        return [Token("("), self.expression, Token(")")]

    def evaluate(self, bindings: Dict[str, Union[int, float, bool, str]]) -> Optional[Union[int, float, bool, str]]:
        return self.expression.evaluate(bindings)

    @property
    def is_key_count(self) -> bool:
        return self.expression.is_key_count

    @property
    def is_count(self) -> bool:
        return self.expression.is_count


class NamedExpression(SqlExpr):
    """An expression with optional name"""

    def __init__(
        self, name: Identifier, expression: EXPR_TYPE
    ) -> None:
        self.name = name
        self.expression = expression

    def column_name(self) -> Identifier:
        if self.name is not None:
            return self.name
        elif type(self.expression) is Column:
            parts = self.expression.name.split(".")
            return parts[0] if len(parts) == 1 else parts[1]
        elif type(self.expression) is AllColumns:
            return Identifier("???")
        else:
            return Identifier("???")

    def type(self) -> str:
        return self.expression.type()

    def sensitivity(self) -> Optional[Union[float, int]]:
        return self.expression.sensitivity()

    def children(self) -> List[Union[Token, ExpressionType]]:
        return [self.expression] + ([Token("AS"), self.name] if self.name is not None else [])

    def evaluate(self, bindings: Dict[str, Union[int, float, bool, str]]) -> Optional[Union[int, float, bool, str]]:
        return self.expression.evaluate(bindings)

    @property
    def is_key_count(self) -> bool:
        return self.expression.is_key_count

    @property
    def is_count(self) -> bool:
        return self.expression.is_count
