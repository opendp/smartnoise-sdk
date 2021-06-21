class StringLiteral:
    def __init__(self, value):
        self.value = value

class NumericLiteral:
    def __init__(self, value):
        self.value = value

class Identifier:
    def __init__(self, name):
        self.name = name

class Attribute:
    def __init__(self, name):
        self.name = name

class RootSelect:
    def __init__(self, target, condition=None):
        self.target = target
        self.condition = condition

class ChildSelect:
    def __init__(self, target, condition=None):
        self.target = target
        self.condition = condition

class DescendantSelect:
    def __init__(self, target, condition=None):
        self.target = target
        self.condition = condition

class AllSelect:
    def __init__(self, condition=None):
        self.condition = condition

class Statement:
    def __init__(self, steps):
        self.steps = steps

class Condition:
    def __init__(self, left : Statement, operator : str, right : Statement):
        self.left = left
        self.right = right
        self.operator = operator

