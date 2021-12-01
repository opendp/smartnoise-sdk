from datetime import date
import calendar

from snsql._ast.tokens import *
import datetime

"""
    date/time processing expressions
"""
def parse_datetime(val):
    if val is None:
        return None
    if isinstance(val, (datetime.date, datetime.time, datetime.datetime)):
        return val
    pass
    parsed = None
    try:
        parsed = datetime.date.fromisoformat(val)
    except:
        pass
    if not parsed:
        try:
            parsed = datetime.time.fromisoformat(val)
        except:
            pass
    if not parsed:
        try:
            parsed = datetime.datetime.fromisoformat(val)
        except:
            pass
    return parsed

class CurrentTimeFunction(SqlExpr):
    def children(self):
        return [Token("CURRENT_TIME")]
    def evaluate(self, bindings):
        return datetime.datetime.now().time()
    def symbol(self, relations):
        return CurrentTimeFunction()

class CurrentDateFunction(SqlExpr):
    def children(self):
        return [Token("CURRENT_DATE")]
    def evaluate(self, bindings):
        return datetime.datetime.now().date()
    def symbol(self, relations):
        return CurrentDateFunction()

class CurrentTimestampFunction(SqlExpr):
    def children(self):
        return [Token("CURRENT_TIMESTAMP")]
    def evaluate(self, bindings):
        return datetime.datetime.now()
    def symbol(self, relations):
        return CurrentTimestampFunction()

class DayNameFunction(SqlExpr):
    def __init__(self, expression):
        self.expression = expression
    def children(self):
        return [Token('DAYNAME'), Token('('), self.expression, Token(')')]
    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        if exp is None:
            return None
        if isinstance(exp, str):
            try:
                exp = datetime.datetime.fromisoformat(exp)
            except:
                pass
        if isinstance(exp, (datetime.date, datetime.datetime)):
            return calendar.day_name[exp.weekday()]
        else:
            raise ValueError(f"Unable to get day name for: {str(exp)}")
    def symbol(self, relations):
        return DayNameFunction(self.expression)

class ExtractFunction(SqlExpr):
    def __init__(self, date_part, expression):
        self.date_part = date_part
        self.expression = expression
    def children(self):
        return [Token("EXTRACT"), Token("("), Token(str(self.date_part).upper()), Token('FROM'), self.expression, Token(")")]
    def evaluate(self, bindings):
        exp = self.expression.evaluate(bindings)
        if exp is None:
            return None
        if isinstance(exp, str):
            parsed = parse_datetime(exp)
        else:
            parsed = exp
        if not isinstance(parsed, (datetime.date, datetime.time, datetime.datetime)):
            raise ValueError(f"Got unknown date/time format: {exp}")
        if self.date_part == 'weekday':
            if isinstance(parsed, datetime.time):
                return None
            return parsed.weekday()
        if self.date_part == 'day':
            if isinstance(parsed, datetime.time):
                return None
            return parsed.day
        elif self.date_part == 'month':
            if isinstance(parsed, datetime.time):
                return None
            return parsed.month
        elif self.date_part == 'year':
            if isinstance(parsed, datetime.time):
                return None
            return parsed.year
        elif self.date_part == 'hour':
            if isinstance(parsed, datetime.date) and not isinstance(parsed, datetime.datetime):
                return 0
            return parsed.hour
        elif self.date_part == 'minute':
            if isinstance(parsed, datetime.date) and not isinstance(parsed, datetime.datetime):
                return 0
            return parsed.minute
        elif self.date_part == 'second':
            if isinstance(parsed, datetime.date) and not isinstance(parsed, datetime.datetime):
                return 0
            return parsed.second
        elif self.date_part == 'microsecond':
            if isinstance(parsed, datetime.date) and not isinstance(parsed, datetime.datetime):
                return 0
            return parsed.microsecond
        else:
            raise ValueError(f"Unknown date part requested: {self.date_part}")
    def symbol(self, relations):
        return ExtractFunction(self.expression.symbol(relations), self.date_part)

