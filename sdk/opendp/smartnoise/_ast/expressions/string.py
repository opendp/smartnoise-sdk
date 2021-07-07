from opendp.smartnoise._ast.tokens import *

class StringAggFunction(SqlExpr):
    def __init__(self, expression, delimiter: str):
        self.expression = expression
        self.delimiter = delimiter
    def children(self):
        return [Token('STRING_AGG'), Token('('), self.expression, Token(','), Token("'"), self.delimiter, Token("'"), Token(')')]
