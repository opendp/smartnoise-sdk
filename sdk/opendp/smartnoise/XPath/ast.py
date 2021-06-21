from typing import Union, Sequence

class StringLiteral:
    def __init__(self, value : str):
        self.value = value

class NumericLiteral:
    def __init__(self, value : Union[int, float]):
        self.value = value

class Identifier:
    def __init__(self, name : str):
        self.name = name

class Attribute:
    def __init__(self, name : str):
        self.name = name

class Statement:
    def __init__(self, steps : Sequence):
        self.steps = steps

class Condition:
    def __init__(self, left : Statement, operator : str, right : Statement):
        self.left = left
        self.right = right
        self.operator = operator

class RootSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition

class ChildSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition

class DescendantSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition

class AllSelect:
    def __init__(self, condition : Condition = None):
        self.condition = condition


