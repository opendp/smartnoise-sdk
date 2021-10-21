from snsql._ast.tokens import *
import datetime

"""
    type processing expressions
"""

class CastFunction(SqlExpr):
    def __init__(self, expression, dbtype):
        self.expression = expression
        self.dbtype = dbtype
    def children(self):
        return [Token('CAST'), Token('('), self.expression, Token('AS'), Token(self.dbtype.upper()), Token(')')]
    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        if exp is None:
            return None
        dbtype = self.dbtype
        if dbtype == "integer":
            return int(exp)
        elif dbtype == "float":
            return float(exp)
        elif dbtype == "boolean":
            return bool(exp)
        elif dbtype == "timestamp":
            return datetime.datetime.fromisoformat(str(exp))
        elif dbtype == "time":
            return datetime.time.fromisoformat(str(exp))
        elif dbtype == "date":
            return datetime.date.fromisoformat(str(exp))
        elif dbtype.startswith("varchar") or dbtype.startswith("char"):
            e = str(exp)
            if dbtype == "varchar" or dbtype == "char":
                return e
            else:
                l = dbtype.index('(')
                r = dbtype.index(')')
                length = r - l
                if len(e) <= length:
                    return e
                else:
                    return e[:length + 1]
        else:
            raise ValueError(f"Unknown type for CAST: {dbtype}")
    def symbol(self, relations):
        return CastFunction(self.expression.symbol(relations), self.dbtype)
        