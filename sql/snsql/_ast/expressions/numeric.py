from numpy.lib.arraysetops import isin
from snsql._ast.tokens import *
import numpy as np
import operator

ops = {
    "+": operator.add,
    "-": operator.sub,
    "/": operator.truediv,
    "*": operator.mul,
    "%": operator.mod,
}

funcs = {
    "abs": np.abs,
    "ceil": np.ceil,
    "ceiling": np.ceil,
    "floor": np.floor,
    "sign": np.sign,
    "sqrt": np.sqrt,
    "square": np.square,
    "exp": np.exp,
    "ln": np.log,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atanh": np.arctanh,
    "degrees": np.degrees,
}

bare_funcs = {
    "pi": lambda: np.pi,
    "rand": lambda: np.random.uniform(),
    "random": lambda: np.random.uniform(),
    "newid": lambda: "-".join(
        [
            "".join([hex(np.random.randint(0, 65535)) for v in range(2)]),
            [hex(np.random.randint(0, 65535))],
            [hex(np.random.randint(0, 65535))],
            [hex(np.random.randint(0, 65535))],
            "".join([hex(np.random.randint(0, 65535)) for v in range(3)]),
        ]
    ),
}


class ArithmeticExpression(SqlExpr):
    """A simple arithmetic expression with left and right side and operator"""

    def __init__(self, left, op, right):
        self.left = left
        self.right = right
        self.op = op

    def type(self):
        if self.op == "/":
            return "float"
        elif self.op == "%":
            return "int"
        elif self.op in ["*", "+", "-"]:
            return min([self.left.type(), self.right.type()])

    def sensitivity(self):
        ls = self.left.sensitivity()
        rs = self.right.sensitivity()
        if rs is None and ls is None:
            return None
        if rs is None and type(self.right) is Literal and self.right.type() in ["int", "float"]:
            if self.op in ["+", "-"]:
                return ls
            elif self.op == "%":
                return self.right.value
            elif self.op == "*":
                return ls * self.right.value
            elif self.op == "/":
                return ls / self.right.value
            else:
                return None
        if ls is None and type(self.left) is Literal and self.left.type() in ["int", "float"]:
            if self.op in ["+", "-"]:
                return rs
            elif self.op == "*":
                return rs * self.left.value
            else:
                return None
        if ls is not None and rs is not None:
            if self.op == "+":
                return ls + rs
            elif self.op == "*":
                return ls * rs
            else:
                return None

    def children(self):
        return [self.left, self.op, self.right]

    def evaluate(self, bindings):
        l = self.left.evaluate(bindings)
        r = self.right.evaluate(bindings)
        if self.op == '/' and int(r) == 0:
            return 0
        else:
            return ops[self.op](l, r)

    def symbol(self, relations):
        return ArithmeticExpression(
            self.left.symbol(relations), self.op, self.right.symbol(relations)
        )


class MathFunction(SqlExpr):
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression

    def symbol_name(self):
        prefix = self.name.lower() + "_"
        return self.prepend(prefix, self.expression.symbol_name())

    def prepend(self, prefix, value):
        # handles generation of a new identifier while honoring escaping rules
        if value == "" or not value[0] in ['"', "`", "["]:
            return prefix + value
        value = value.replace("`", '"').replace("[", '"').replace("]", '"')
        parts = value.split('"')
        if len(parts) == 3:
            return '"' + prefix + parts[1] + '"'
        else:
            return prefix + "_x_" + value.replace('"', "").replace(" ", "")

    def children(self):
        return [self.name, Token("("), self.expression, Token(")")]

    def type(self):
        return "float"

    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        return funcs[self.name.lower()](exp)

    def symbol(self, relations):
        return MathFunction(self.name, self.expression.symbol(relations))


class PowerFunction(SqlExpr):
    def __init__(self, expression, power):
        self.expression = expression
        self.power = power

    def children(self):
        return [Token("POWER"), Token("("), self.expression, Token(","), self.power, Token(")")]

    def type(self):
        return self.expression.type()

    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        return np.power(exp, self.power.value)

    def symbol(self, relations):
        return PowerFunction(self.expression.symbol(relations), self.power.symbol(relations))


class BareFunction(SqlExpr):
    def __init__(self, name):
        self.name = name

    def children(self):
        return [self.name, Token("("), Token(")")]

    def evaluate(self, bindings):
        vec = bindings[list(bindings.keys())[0]]  # grab the first column
        return [bare_funcs[self.name.lower()]() for v in vec]


class RoundFunction(SqlExpr):
    def __init__(self, expression, decimals):
        if not isinstance(decimals.value, int):
            raise ValueError("Decimals argument must be integer")
        self.expression = expression
        self.decimals = decimals

    def children(self):
        start = [Token("ROUND"), Token("("), self.expression]
        end = [Token(")")]
        middle = [] if not self.decimals else [Token(","), self.decimals]
        return start + middle + end

    def evaluate(self, bindings):
        decimals = self.decimals.evaluate(bindings)
        exp = self.expression.evaluate(bindings)
        return np.round(exp, decimals if decimals else 0)

    def symbol(self, relations):
        return RoundFunction(self.expression.symbol(relations), self.decimals)


class TruncFunction(SqlExpr):
    def __init__(self, expression, decimals):
        if not isinstance(decimals.value, int):
            raise ValueError("Decimals argument must be integer")
        self.expression = expression
        self.decimals = decimals
    def children(self):
        start = [Token("TRUNCATE"), Token("("), self.expression]
        end = [Token(")")]
        middle = [] if not self.decimals else [Token(","), self.decimals]
        return start + middle + end
    def evaluate(self, bindings):
        decimals = self.decimals.evaluate(bindings)
        exp = self.expression.evaluate(bindings)
        if decimals == None:
            decimals = 0
        shift = float(10 ** decimals)
        v = float(exp * shift)
        v = np.floor(v)
        v = v / shift
        if isinstance(exp, int):
            v = int(v)
        return v
    def symbol(self, relations):
        return TruncFunction(self.expression.symbol(relations), self.decimals)
