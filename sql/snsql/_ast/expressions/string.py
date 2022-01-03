from snsql._ast.tokens import *

"""
    string processing expressions
"""

class LowerFunction(SqlExpr):
    def __init__(self, expression):
        self.expression = expression
    def children(self):
        return [Token("LOWER"), Token("("), self.expression, Token(")")]
    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        return str(exp).lower()
    def symbol(self, relations):
        return LowerFunction(self.expression.symbol(relations))

class UpperFunction(SqlExpr):
    def __init__(self, expression):
        self.expression = expression
    def children(self):
        return [Token("UPPER"), Token("("), self.expression, Token(")")]
    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        return str(exp).upper()
    def symbol(self, relations):
        return UpperFunction(self.expression.symbol(relations))

class TrimFunction(SqlExpr):
    def __init__(self, expression):
        self.expression = expression
    def children(self):
        return [Token("TRIM"), Token("("), self.expression, Token(")")]
    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        if exp is None:
            return None
        else:
            return str(exp).strip()
    def symbol(self, relations):
        return TrimFunction(self.expression.symbol(relations))

class CharLengthFunction(SqlExpr):
    def __init__(self, expression):
        self.expression = expression
    def children(self):
        return [Token("CHAR_LENGTH"), Token("("), self.expression, Token(")")]
    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        if exp is None:
            return None
        else:
            return len(str(exp))
    def symbol(self, relations):
        return TrimFunction(self.expression.symbol(relations))

class PositionFunction(SqlExpr):
    def __init__(self, search, source):
        self.search = search
        self.source = source
    def children(self):
        return [Token("POSITION"), Token("("), self.search, Token("IN"), self.source, Token(")")]
    def evaluate(self, bindings):
        search = self.search.evaluate(bindings)
        source = self.source.evaluate(bindings)
        if search is None or source is None:
            return None
        if search not in source:
            return 0
        else:
            return source.index(search) + 1
    def symbol(self, relations):
        return PositionFunction(self.search.symbol(relations), self.source.symbol(relations))

class SubstringFunction(SqlExpr):
    def __init__(self, source, start, length):
        self.source = source
        self.start = start
        self.length = length
    def children(self):
        start = [Token("SUBSTRING"), Token("("), self.source, Token('FROM'), self.start]
        middle = [] if self.length is None else [Token('FOR'), self.length]
        end = [Token(")")]
        return start + middle + end
    def evaluate(self, bindings):
        source = self.source.evaluate(bindings) if self.source else None
        start = self.start.evaluate(bindings) if self.start else None
        length = self.length.evaluate(bindings) if self.length else None
        if source is None or start is None:
            return None
        if not isinstance(start, int):
            raise ValueError(f"Start position must evaluate to an integer: {str(self.start)}")
        source = str(source)
        if start == 0:
            start = 1
        if start > 0:
            start = start - 1  # python indices start at 0
        if start < 0:
            start = len(source) + start
        if not length:
            return source[start:]
        else:
            if not isinstance(length, int):
                raise ValueError("Substring length if provided must be positive")
            if length < 1:
                return None
            return source[start:start+length]
    def symbol(self, relations):
        return SubstringFunction(self.source.symbol(relations), self.start.symbol(relations), self.length.symbol(relations))

class ConcatFunction(SqlExpr):
    def __init__(self, expressions):
        self.expressions = expressions
    def children(self):
        return [Token("CONCAT"), Token("("), Seq(self.expressions), Token(")")]
    def evaluate(self, bindings):
        eval = [e.evaluate(bindings) for e in self.expressions]
        if any([v is None for v in eval]):
            return None
        return ''.join([str(e) for e in eval])
    def symbol(self, relations):
        symbols = [e.symbol(relations) for e in self.expressions]
        return ConcatFunction(symbols)

class CoalesceFunction(SqlExpr):
    def __init__(self, expressions):
        self.expressions = expressions
    def children(self):
        return [Token("COALESCE"), Token("("), Seq(self.expressions), Token(")")]
    def evaluate(self, bindings):
        eval = [e.evaluate(bindings) for e in self.expressions]
        eval = [e for e in eval if e is not None]
        if len(eval) == 0:
            return None
        else:
            return eval[0]
    def symbol(self, relations):
        symbols = [e.symbol(relations) for e in self.expressions]
        return CoalesceFunction(symbols)

