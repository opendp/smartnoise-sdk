from opendp.smartnoise._ast.tokens import *

class CastFunction(SqlExpr):
    def __init__(self, expression, typename: str):
        self.expression = expression
        self.typename = typename
    def children(self):
        return [Token('CAST'), Token('('), self.expression, Token('AS'), self.typename, Token(')')]