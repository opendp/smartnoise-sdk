from opendp.smartnoise._ast.tokens import *

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from opendp.smartnoise._ast.types_ast import ExpressionType, BooleanExpressionType, TableColumn

"""
    SQL-specific expressions
"""


class AllColumns(SqlExpr):
    """A SELECT with * or Table.*"""

    def __init__(self, table: Optional[Literal]=None):
        self.table = table

    def __str__(self):
        return (self.table + "." if self.table is not None else "") + "*"

    def __hash__(self):
        return hash(str(self))

    def all_symbols(self, relations: List[SqlRel]) -> Optional[TableColumn]:
        sym = [r.all_symbols(self) for r in relations if r.alias_match(str(self))]
        if len(sym) == 0:
            raise ValueError("Column cannot be found " + str(self))
        return flatten(sym)

    @property
    def is_key_count(self):
        return True

    @property
    def is_count(self):
        return True


class AggFunction(SqlExpr):
    """A function such as SUM, COUNT, AVG"""

    def __init__(self, name: FuncName, quantifier: Token, expression: ExpressionType):
        self.name = name
        self.quantifier = quantifier
        self.expression = expression

    def symbol_name(self) -> str:
        prefix = (
            self.name.lower()
            + "_"
            + ("" if self.quantifier is None else self.quantifier.lower() + "_")
        )
        return self.prepend(prefix, self.expression.symbol_name())

    def prepend(self, prefix: str, value: str) -> str:
        # handles generation of a new identifier while honoring escaping rules
        if value == "" or not value[0] in ['"', "`", "["]:
            return prefix + value
        value = value.replace("`", '"').replace("[", '"').replace("]", '"')
        parts = value.split('"')
        if len(parts) == 3:
            return '"' + prefix + parts[1] + '"'
        else:
            return prefix + "_x_" + value.replace('"', "").replace(" ", "")

    def is_aggregate(self) -> bool:
        return self.name in ["SUM", "COUNT", "MIN", "MAX", "AVG", "VAR"]

    def symbol(self, relations) -> 'AggFunction':
        return AggFunction(self.name, self.quantifier, self.expression.symbol(relations))

    def type(self):
        # will switch to lookup table
        if self.name == "SUM":
            return self.expression.type()
        elif self.name == "COUNT":
            return "int"
        elif self.name == "MIN":
            return self.expression.type()
        elif self.name == "MAX":
            return self.expression.type()
        elif self.name in ["VAR", "VARIANCE", "AVG", "STD", "STDDEV"]:
            return "float"
        else:
            return "unknown"

    def sensitivity(self):
        # will switch to lookup table
        if self.name == "SUM":
            return self.expression.sensitivity()
        elif self.name == "COUNT":
            return 1
        elif self.name == "AVG":
            return self.expression.sensitivity()
        elif self.name == "MIN":
            return self.expression.sensitivity()
        elif self.name == "MAX":
            return self.expression.sensitivity()
        else:
            return None

    def children(self):
        return [self.name, Token("("), self.quantifier, self.expression, Token(")")]

    def evaluate(self, bindings):
        # need to decide what to do with this
        return self.expression.evaluate(bindings)

    @property
    def is_key_count(self):
        if self.name == "SUM":
            return self.expression.is_key_count
        elif self.name == "COUNT":
            return self.expression.is_key_count
        else:
            return False

    @property
    def is_count(self):
        if self.name == "SUM":
            return self.expression.is_count
        elif self.name == "COUNT":
            return True
        else:
            return False


class RankingFunction(SqlExpr):
    def __init__(self, name: FuncName, over: 'OverClause'):
        self.name = name
        self.over = over

    def children(self):
        return [self.name, Token("("), Token(")"), self.over]

    def symbol(self, relations):
        return RankingFunction(self.name, self.over.symbol(relations))


class OverClause(SqlExpr):
    def __init__(self, partition: ExpressionType, order: 'Order'):
        self.partition = partition
        self.order = order

    def children(self):
        pre = [Token("OVER"), Token("(")]
        post = [Token(")")]
        partition = (
            [] if self.partition is None else [Token("PARTITION"), Token("BY"), self.partition]
        )
        order = [] if self.order is None else [self.order]
        return pre + partition + order + post

    def symbol(self, relations):
        return OverClause(self.partition.symbol(relations), self.order.symbol(relations))


class GroupingExpression(SqlExpr):
    """An expression used in Group By"""

    def __init__(self, expression: ExpressionType):
        self.expression = expression

    def type(self):
        return self.expression.type()

    def children(self):
        return [self.expression]

    def symbol(self, relations):
        return GroupingExpression(self.expression.symbol(relations))


class SortItem(SqlExpr):
    """Used to sort a query's output"""

    def __init__(self, expression: ExpressionType, order: 'Order'):
        self.expression = expression
        self.order = order

    def type(self):
        return self.expression.type()

    def children(self):
        return [self.expression, self.order]

    def symbol(self, relations):
        return SortItem(
            self.expression.symbol(relations),
            None if self.order is None else self.order.symbol(relations),
        )


class BooleanJoinCriteria(SqlExpr):
    """Join criteria using boolean expression"""

    def __init__(self, expression: BooleanExpressionType):
        self.expression = expression

    def children(self):
        return [Token("ON"), self.expression]

    def symbol(self, relations):
        return BooleanJoinCriteria(self.expression.symbol(relations))


class UsingJoinCriteria(SqlExpr):
    """Join criteria with USING syntax"""

    def __init__(self, identifiers: List[Identifier]):
        self.identifiers = Seq(identifiers)

    def children(self):
        return [Token("USING"), Token("("), self.identifiers, Token(")")]

    def symbol(self, relations):
        return UsingJoinCriteria(self.identifiers.symbol(relations))
