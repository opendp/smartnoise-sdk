from typing import Union, Sequence, Iterable
import operator
import numpy as np
from numpy.lib.arraysetops import isin
from pandas.core.indexing import IndexSlice

ops = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "=": operator.eq,
    "!=": operator.ne,
    "==": operator.eq,
    "<>": operator.ne,
    "and": np.logical_and,
    "or": np.logical_or,
}

def flatten(lst):
    for item in lst:
        if isinstance(item,Iterable) and not isinstance(item,str):
            yield from flatten(item)
        else:
            yield item

class Literal:
    def __str__(self):
        return str(self.value)
    def evaluate(self, node, idx):
        return self.value

class StringLiteral(Literal):
    def __init__(self, value : str):
        self.value = value
    def __str__(self):
        return "'" + self.value + "'"

class NumericLiteral(Literal):
    def __init__(self, value : Union[int, float]):
        self.value = value

class NullLiteral:
    def __init__(self):
        self.value = None
    def __str__(self):
        return 'NULL'

class Identifier:
    def __init__(self, name : str):
        self.name = name
    def __str__(self):
        return self.name

class AllNodes:
    def __str__(self):
        return '*'
    def evaluate(self, node, idx):
        node = traverse_short(node)
        return [c for c in node.children() if c is not None]

class AllAttributes:
    def __str__(self):
        return '@*'
    def evaluate(self, node, idx):
        node = traverse_short(node)
        d = node.__dict__
        return [Attribute(k, d[k]) for k in d.keys() if d[k] is not None]

class Attribute:
    def __init__(self, name : str, value : str = None):
        self.name = name
        self.value = value
    def __str__(self):
        return "@" + self.name
    def children(self):
        return [self.value]

class Statement:
    def __init__(self, steps : Sequence):
        self.steps = steps
    def __str__(self):
        out = str(self.steps[0])
        if len(self.steps) > 1:
            for step in self.steps[1:]:
                if not isinstance(step, IndexSelector):
                    out = out + '/'
                out = out + str(step)
        return out
    def evaluate(self, node, idx=0):
        n = [traverse_short(node)]
        for s in self.steps:
            x = 1
            res = list(flatten([s.evaluate(nn, idx) for nn, idx in zip(n, range(len(n)))]))
            n = [r for r in res if r is not None]
        return [nn for nn in n if nn is not None]

class Condition:
    def __init__(self, left : Statement, operator : str, right : Statement):
        self.left = left
        self.right = right
        self.operator = operator
    def __str__(self):
        return '[' + str(self.left) + ('' if self.operator is None else ' ' + self.operator + ' ' + str(self.right)) + ']'
    def evaluate(self, node, idx):
        node = traverse_short(node)
        l = self.left.evaluate(node, 0)
        if self.operator is None:
            if isinstance(l, int):
                return node.children()[l]
            else:
                if len(l) >= 1:
                    return node
                else:
                    return None
        else:
            r = self.right.evaluate(node, 0)
            if isinstance(l, list) and not isinstance(l, str):
                if len(l) != 1:
                    return None
                l = l[0]
            if isinstance(r, list) and not isinstance(r, str):
                if len(r) !=1:
                    return None
                r = r[0]
            l = traverse(l)
            r = traverse(r)
            match = bool(ops[self.operator.lower()](l, r))
            return node if match else None

class RootSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def __str__(self):
        return '/' + str(self.target) + ('' if self.condition is None else str(self.condition))
    def evaluate(self, node, idx):
        r = []
        node = traverse_short(node)
        if isinstance(self.target, Identifier):
            cname = node.__class__.__name__
            if self.target.name == cname:
                r.append(node)
        elif isinstance(self.target, Attribute): # it's an attribute
            if hasattr(node, self.target.name):
                v = getattr(node, self.target.name)
                if v is not None:
                    r.append(Attribute(self.target.name, v))
        if r != [] and self.condition is not None:
            r = [self.condition.evaluate(node, 0)]
        return r

class ChildSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def __str__(self):
        return str(self.target) + ('' if self.condition is None else str(self.condition))
    def evaluate(self, node, idx):
        r = []
        if isinstance(self.target, Identifier):
            for n in node.children():
                while isinstance(n, (Attribute, Literal)) or n.__class__.__name__ == 'Literal':
                    n = n.value            
                cname = n.__class__.__name__
                if self.target.name == cname:
                    r.append(n)
        elif isinstance(self.target, Attribute): # it's an attribute
            while isinstance(node, (Attribute, Literal)) or node.__class__.__name__ == 'Literal':
                node = node.value            
            if hasattr(node, self.target.name):
                v = getattr(node, self.target.name)
                if v is not None:
                    r.append(Attribute(self.target.name, v))
        elif isinstance(self.target, (AllNodes, AllAttributes)):
            r.append(list(flatten(self.target.evaluate(node, 0))))
        if r != [] and self.condition is not None:
            r = [self.condition.evaluate(n, idx) for n, idx in zip(r, range(len(r)))]
            r = [n for n in r if n is not None]
        return r

class DescendantSelect:
    # DescendantSelector over child nodes
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def __str__(self):
        return '/' + str(self.target) + ('' if self.condition is None else str(self.condition))
    def evaluate(self, node, idx):
        r = []
        node = traverse_short(node)
        # recurse descendants
        step = RootDescendantSelect(self.target, self.condition) # should condition here be None?
        children = node.children()
        for n, idx in zip(children, range(len(children))):
            if n is not None:
                r.append(list(flatten(step.evaluate(n, idx))))
        return flatten(r)

class RootDescendantSelect:
    def __init__(self, target : Union[Identifier, Attribute], condition : Condition = None):
        self.target = target
        self.condition = condition
    def __str__(self):
        return '//' + str(self.target) + ('' if self.condition is None else str(self.condition))
    def evaluate(self, node, idx):
        r = []
        node = traverse_short(node)
        # first look at self
        if isinstance(self.target, Identifier):
            cname = node.__class__.__name__
            if self.target.name == cname:
                r.append(node)
        elif isinstance(self.target, Attribute): # it's an attribute
            if hasattr(node, self.target.name):
                v = getattr(node, self.target.name)
                if v is not None:
                    r.append(Attribute(self.target.name, v))
        elif isinstance(self.target, (AllNodes, AllAttributes)):
            r.append(list(flatten(self.target.evaluate(node, 0))))
        # recurse descendants
        children = node.children()
        
        for n, idx in zip(children, range(len(children))):
            if n is not None:
                r.append(list(flatten(self.evaluate(n, idx))))
        r = list(flatten(r))
        if r != [] and self.condition is not None:
            r = [self.condition.evaluate(n, idx) for n , idx in zip(r, range(len(r)))]
            r = [n for n in r if n is not None]
        return list(flatten(r))

class IndexSelector:
    def __init__(self, index):
        self.index = index
    def __str__(self):
        return '[' + str(self.index) + ']'
    def evaluate(self, node, idx):
        return node if idx == self.index else None

def traverse_short(node):
    # traverse XPath Attributes and Literals
    while isinstance(node, (Attribute, Literal)):
        node = node.value
    return node

def traverse(node):
    # traverse all literals to prepare for boolean
    while isinstance(node, (Attribute, Literal)) or node.__class__.__name__ == 'Literal':
        node = node.value
    return node
